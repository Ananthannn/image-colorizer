import os


def load_images(data_dir):
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(supported)]

    if len(files) == 0:
        raise ValueError(f"No images found in {data_dir}")

    print(f"Found {len(files)} images. Loading...")
    return files