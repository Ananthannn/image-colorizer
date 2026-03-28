import os
import cv2
import numpy as np
from skimage.color import rgb2lab

def preprocess_images(data_dir, files, image_size=256):
    L_inputs = []
    AB_targets = []

    for filename in files:
        path = os.path.join(data_dir, filename)

        # Read image in BGR (OpenCV default), convert to RGB
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"  Skipping unreadable file: {filename}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to fixed size
        img_rgb = cv2.resize(img_rgb, (image_size, image_size))

        # Normalize to [0, 1] for skimage
        img_rgb_norm = img_rgb.astype(np.float32) / 255.0

        # Convert RGB → LAB
        img_lab = rgb2lab(img_rgb_norm)

        # Extract channels
        L  = img_lab[:, :, 0]          # Shape: (256, 256)   Range: [0, 100]
        AB = img_lab[:, :, 1:]         # Shape: (256, 256, 2) Range: [-128, 127]

        # L channel stays raw [0, 100] — the model normalizes internally via x / 100.0
        L_norm = L

        # Normalize AB to [-1, 1]
        AB_norm = AB / 128.0

        # Add channel dimension to L: (256, 256) → (256, 256, 1)
        L_inputs.append(L_norm[:, :, np.newaxis])
        AB_targets.append(AB_norm)

    X = np.array(L_inputs,   dtype=np.float32)  # (N, 256, 256, 1)
    Y = np.array(AB_targets, dtype=np.float32)  # (N, 256, 256, 2)

    print(f"Dataset shape: X={X.shape}, Y={Y.shape}")
    return X, Y