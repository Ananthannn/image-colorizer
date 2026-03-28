import os
import sys
import argparse
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb

# Add parent directory to access PyTorch architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.architecture import build_colorization_model

def predict_color(image_path, model_weights_path):
    print(f"Loading image from '{image_path}'...")
    
    if not os.path.exists(image_path):
        print("❌ Image file not found!")
        return
        
    if not os.path.exists(model_weights_path):
        print(f"❌ Custom Weights not found at '{model_weights_path}'. Please train your model first!")
        return

    # Image preparation (Just like training)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("❌ OpenCV could not read the image")
        return
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]
    
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_norm = img_resized.astype(np.float32) / 255.0
    
    img_lab = rgb2lab(img_norm)
    L_channel = img_lab[:, :, 0] # Range 0 to 100
    
    # Format to (1 Batch, 1 Channel, 256 Height, 256 Width) for PyTorch
    model_input_np = L_channel[np.newaxis, np.newaxis, :, :]
    
    # Cast to PyTorch Tensor structure
    model_input = torch.tensor(model_input_np, dtype=torch.float32)

    # Boot up hardware architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_colorization_model().to(device)
    
    # Load your `.pth` parameters into the network directly mapping any device mismatch 
    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    
    # Disable dropout/batchnorm instability for inference
    model.eval()

    print(f"🎨 Colorizing image dynamically on {device.type.upper()}...")
    
    # Predict without generating gradient memory
    with torch.no_grad():
        # Inject Tensor into computing buffer directly
        ab_predictions = model(model_input.to(device))
    
    # Bring memory back down to CPU standard numpy format
    ab_predictions_np = ab_predictions.cpu().numpy()
    
    # Swap shape from standard PyTorch (1, 2, 256, 256) back to OpenCv standard (256, 256, 2)
    AB_channel = ab_predictions_np[0].transpose(1, 2, 0)
    
    # Undo PyTorch's internal Tanh limits (-1 to 1) out to LAB real colors (-128, 127)
    AB_channel = AB_channel * 128.0

    print("Merging generated colors back together...")
    out_lab = np.zeros((256, 256, 3), dtype=np.float32)
    out_lab[:, :, 0] = L_channel
    out_lab[:, :, 1:] = AB_channel

    out_rgb = lab2rgb(out_lab) # Yields 0-1
    out_rgb_original_size = cv2.resize(out_rgb, (original_w, original_h))
    
    # Cast float values to standard 8-bit monitor pixels (0-255)
    out_rgb_final = (out_rgb_original_size * 255).astype(np.uint8)
    
    # Standardize output for Windows/OpenCV writing system
    out_bgr_final = cv2.cvtColor(out_rgb_final, cv2.COLOR_RGB2BGR)

    output_filename = "colorized_" + os.path.basename(image_path)
    cv2.imwrite(output_filename, out_bgr_final)
    print(f"✅ Masterpiece Created! Painted image saved as: '{output_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use AI Model to paint grayscale images")
    parser.add_argument("image", help="Absolute or relative path to your grayscale photo")
    parser.add_argument("--weights", default="../checkpoints/colorizer_weights.pth", help="Path to your trained .pth weights file")
    args = parser.parse_args()
    
    predict_color(args.image, args.weights)
