import os
import sys
import urllib.request
import zipfile
import time

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
import numpy as np

from data.image_load.load_image import load_images
from data.image_preprocess.preprocess_image import preprocess_images
from model.architecture import build_colorization_model


def download_coco_val2017(dataset_dir):
    """Automatically downloads COCO val2017 with progress and corruption checks."""
    os.makedirs(os.path.dirname(dataset_dir), exist_ok=True)
    zip_path = os.path.join(os.path.dirname(dataset_dir), "val2017.zip")
    url = "http://images.cocodataset.org/zips/val2017.zip"
    expected_min_size = 750 * 1024 * 1024  # ~750MB minimum for valid zip
    
    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 100:
        return  # Dataset already extracted and looks valid
    
    # Check if existing zip is corrupt (too small = failed download)
    if os.path.exists(zip_path):
        actual_size = os.path.getsize(zip_path)
        if actual_size < expected_min_size:
            print(f"⚠️  Found corrupt zip ({actual_size / (1024*1024):.1f} MB — expected ~800 MB). Re-downloading...")
            os.remove(zip_path)
    
    # Download with progress
    if not os.path.exists(zip_path):
        print("⏳ Downloading COCO 2017 Validation Set (~800 MB)...")
        
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   Downloading: {mb_done:.0f}/{mb_total:.0f} MB ({pct:.1f}%)", end='', flush=True)
        
        try:
            urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
            print()  # newline after progress
        except Exception as e:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise RuntimeError(f"Download failed: {e}. Check your internet connection and try again.")
        
        # Validate download size
        actual_size = os.path.getsize(zip_path)
        if actual_size < expected_min_size:
            os.remove(zip_path)
            raise RuntimeError(f"Download incomplete ({actual_size / (1024*1024):.1f} MB). Try again.")
    
    # Extract with validation
    print("📦 Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(dataset_dir))
    except zipfile.BadZipFile:
        os.remove(zip_path)
        raise RuntimeError("Zip file is corrupt. Deleted it — run the script again to re-download.")

def main():
    # 1. Configuration Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_storage = os.path.join(base_dir, "dataset")
    dataset_dir = os.path.join(dataset_storage, "val2017")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    os.makedirs(dataset_storage, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    batch_size = 16
    epochs = 10
    learning_rate = 1e-3

    print("🚀 Initializing PyTorch Colorizer Training pipeline...")

    # 2. Hardware Acceleration Check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"\n🚀 GPU DETECTED: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN enabled: {torch.backends.cudnn.enabled}")
        
        # Enable cuDNN auto-tuner — finds the fastest convolution algorithms
        # for the fixed 256×256 input size. Big speedup for repeated training.
        torch.backends.cudnn.benchmark = True
        print(f"   cuDNN benchmark: ENABLED ✅")
        
        # Check if AMP (Automatic Mixed Precision) is viable
        use_amp = True
        print(f"   Mixed Precision (FP16): ENABLED ✅ — using tensor cores for ~2× speed\n")
    else:
        use_amp = False
        print("\n⚠️ WARNING: GPU not detected! Training will be slow on CPU.\n")

    # 3. Download & Check Data
    download_coco_val2017(dataset_dir)
    print(f"📂 Dataset ready at: {dataset_dir}")

    # 4. Process Images into memory
    print("Pre-processing Images into PyTorch tensors...")
    try:
        files = load_images(dataset_dir)
        # Using a slice of the dataset to avoid RAM blowout. Adjust if you have 64GB RAM!
        files_subset = files[:5000] 
        X_train, Y_train = preprocess_images(dataset_dir, files_subset, image_size=256)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # PyTorch requires dimensions (Batch, Channels, Height, Width).
    # Numpy images are loaded as (Batch, Height, Width, Channels).
    X_train_pt = np.transpose(X_train, (0, 3, 1, 2))
    Y_train_pt = np.transpose(Y_train, (0, 3, 1, 2))

    dataset = TensorDataset(torch.tensor(X_train_pt, dtype=torch.float32), 
                            torch.tensor(Y_train_pt, dtype=torch.float32))
    
    # pin_memory=True pre-stages data in page-locked RAM for faster GPU transfer
    # num_workers=4 loads batches in parallel background threads
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        num_workers=4,
        persistent_workers=True
    )

    # Free the numpy arrays — data now lives in the TensorDataset
    del X_train, Y_train, X_train_pt, Y_train_pt

    # 5. Build Model and Optimizer
    model = build_colorization_model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # GradScaler for mixed precision — prevents FP16 gradients from underflowing
    scaler = GradScaler('cuda', enabled=use_amp)

    # Report GPU memory after model is loaded
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        print(f"📊 GPU memory used by model: {allocated:.0f} MB")

    best_loss = float('inf')
    best_pth = os.path.join(checkpoint_dir, "colorizer_weights.pth")

    # 6. Training Loop with AMP
    print(f"🔥 Starting Training for {epochs} Epochs on {device.type.upper()}...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        
        batch_count = len(dataloader)
        start_time = time.time()
        
        # Loop over batches
        for i, (inputs, targets) in enumerate(dataloader):
            # Send data to GPU
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Automatic Mixed Precision forward pass
            with autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # AMP-aware backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Track metrics
            epoch_loss += loss.item()
            with torch.no_grad():
                mae = torch.mean(torch.abs(outputs - targets)).item()
            epoch_mae += mae
            
            if i % 10 == 0 or i == batch_count - 1:
                print(f"  Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{batch_count}] - Loss: {loss.item():.4f} - MAE: {mae:.4f}", end='\r')
        
        # Calculate Epoch Statistics
        avg_loss = epoch_loss / batch_count
        avg_mae = epoch_mae / batch_count
        elapsed = time.time() - start_time
        
        # GPU memory tracking
        mem_info = ""
        if device.type == 'cuda':
            mem_used = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
            mem_info = f" | Peak GPU Mem: {mem_used:.0f} MB"
        
        print(f"\n✅ Epoch [{epoch+1}/{epochs}] finished in {elapsed:.1f}s | Avg Loss: {avg_loss:.4f} | Avg MAE: {avg_mae:.4f}{mem_info}")
        
        # Automatic Checkpoint saving
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_pth)
            print(f"   💾 Saved best model! (Loss dropped to {best_loss:.4f})")
            
    print("\n🏁 Training Complete!")

if __name__ == "__main__":
    main()
