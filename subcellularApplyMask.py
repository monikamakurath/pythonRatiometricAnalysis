import os
import numpy as np
import tifffile as tiff

# Try importing czifile (only needed for .czi files)
try:
    import czifile
    CZIFILE_AVAILABLE = True
except ImportError:
    CZIFILE_AVAILABLE = False
    print("czifile not found. You will only be able to load .tif files.")

def preprocess_image(image):
    image = np.squeeze(image)
    shape = image.shape
    num_dims = len(shape)
    if num_dims == 2:
        image = np.expand_dims(image, axis=(0, 1))
    elif num_dims == 3:
        if shape[0] in [2, 3, 4]:
            image = np.expand_dims(image, axis=0)
        else:
            image = np.expand_dims(image, axis=1)
    elif num_dims == 4:
        if shape[1] > 10:
            image = np.max(image, axis=1, keepdims=True)
    elif num_dims == 5:
        image = np.max(image, axis=2)
    print(f"Processed shape: {image.shape}")
    return image

def apply_mask_to_stack(image, mask_stack):
    green_image = image[:, 0, :, :]
    red_image = image[:, 1, :, :]
    masked_green = green_image * (mask_stack == 0)
    masked_red = red_image * (mask_stack == 0)
    return np.stack((masked_green, masked_red), axis=1)

def save_masked_stack(masked_stack, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"masked_stack_{label}.tiff")
    tiff.imwrite(output_file, masked_stack.astype(np.uint16), photometric="minisblack", planarconfig="separate", metadata={'axes': 'TCYX'})
    print(f"Masked stack saved at {output_file}")

def subcellularApplyMask(file_path, output_dir):
    if file_path.endswith(".czi"):
        if not CZIFILE_AVAILABLE:
            print("Error: czifile module not installed. Cannot process .czi files.")
            return
        image = czifile.imread(file_path)
    elif file_path.endswith(".tif") or file_path.endswith(".tiff"):
        image = tiff.imread(file_path)
        print(f"Loaded TIFF shape: {image.shape}")
    else:
        print("Unsupported file format. Please select a .czi or .tif file.")
        return

    image = preprocess_image(image)
    base_path = os.path.splitext(file_path)[0]

    for label in ["", "-nucleus", "-periphery"]:
        mask_file_path = base_path + f"{label}-masks.tif"
        if not os.path.exists(mask_file_path):
            print(f"Warning: Mask file not found at {mask_file_path}. Skipping.")
            continue

        mask_stack = tiff.imread(mask_file_path)
        if mask_stack.shape != image[:, 0, :, :].shape:
            print(f"Error: Mask and image dimensions do not match for {label}! {mask_stack.shape} vs {image[:, 0, :, :].shape}")
            continue

        masked_stack = apply_mask_to_stack(image, mask_stack)
        save_masked_stack(masked_stack, output_dir, label if label else "whole")
