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
    """
    Preprocess fluorescence images while handling different shapes dynamically.

    Args:
        image (np.ndarray): The loaded fluorescence image array.

    Returns:
        np.ndarray: Processed image with standard shape (T, C, Y, X).
    """

    # Remove singleton dimensions
    #print("Shape before squeeze: ", image.shape)
    image = np.squeeze(image)
    #print("Shape after squeeze: ", image.shape)

    # Determine new shape
    shape = image.shape

    num_dims = len(shape)

    # Standardize the shape (assume at least T, C, Y, X structure)
    if num_dims == 2:  # Likely (Y, X)
        image = np.expand_dims(image, axis=(0, 1))  # Add time and channel axes
    elif num_dims == 3:  # (Frames, Y, X) or (C, Y, X)
        if shape[0] in [2, 3, 4]:  # Likely a channel dimension
            image = np.expand_dims(image, axis=0)  # Add time axis
        else:  # Likely (T, Y, X)
            image = np.expand_dims(image, axis=1)  # Add channel axis
    elif num_dims == 4:  # (T, C, Y, X) or (C, Z, Y, X)
        if shape[1] > 10:  # If the second dimension is large, it is likely Z-stack
            image = np.max(image, axis=1, keepdims=True)  # Max project across Z
    elif num_dims == 5:  # (T, C, Z, Y, X)
        image = np.max(image, axis=2)  # Max project across Z

    print(f"Processed shape: {image.shape}")
    return image


def applyMask(file_path, output_dir):
    """Applies a binary mask to a multi-channel image stack and saves the result."""

    # Load image (CZI or TIFF)
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

    # Process image to standard shape (T, C, Y, X)
    image = preprocess_image(image)

    # Ensure at least 2 channels are available
    if image.shape[1] < 2:
        print(f"Error: Expected at least 2 channels but found {image.shape[1]}.")
        return

    # Extract the green and red channels
    green_image = image[:, 0, :, :]  # Shape: (T, Y, X)
    red_image = image[:, 1, :, :]  # Shape: (T, Y, X)

    # Load mask file
    mask_file_path = os.path.splitext(file_path)[0] + "-masks.tif"
    if not os.path.exists(mask_file_path):
        print(f"Error: Mask file not found at {mask_file_path}")
        return
    mask_stack = tiff.imread(mask_file_path)

    # Ensure mask shape matches (T, Y, X)
    if mask_stack.shape != green_image.shape:
        print(f"Error: Mask and image dimensions do not match! {mask_stack.shape} vs {green_image.shape}")
        return

    # Apply the mask (keep only where mask is black)
    masked_green = green_image * (mask_stack == 0)
    masked_red = red_image * (mask_stack == 0)

    # Combine the masked channels
    masked_stack = np.stack((masked_green, masked_red), axis=1)  # Shape: (T, 2, Y, X)

    #print("Masked stack final shape before save: ", masked_stack.shape)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the masked stack
    output_file_masked_stack = os.path.join(output_dir, "masked_stack.tiff")
    tiff.imwrite(output_file_masked_stack, masked_stack.astype(np.uint16),
                    photometric="minisblack", planarconfig="separate", metadata={'axes': 'TCYX'})

    print(f"Masked stack saved at {output_file_masked_stack}")

    return output_file_masked_stack
