import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tkinter import Tk, filedialog

# Try importing czifile (only needed for .czi files)
try:
    import czifile

    CZIFILE_AVAILABLE = True
except ImportError:
    CZIFILE_AVAILABLE = False
    print("Czifile not found, loading .tif instead.")


def loadFile():
    # Prompt user to select a file (.czi or .tif)
    Tk().withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("CZI files", "*.czi"), ("TIFF files", "*.tif"), ("All files", "*.*")]
    )

    if not file_path:
        print("No file selected.")
        return None, None

    # Generate an output directory name
    file_name = os.path.basename(file_path)
    file_base_name = os.path.splitext(file_name)[0]  # Remove .czi or .tif extension
    output_dir = os.path.join(os.path.dirname(file_path), f"{file_base_name}_output")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables
    green_channel, red_channel = None, None

    # **Handle CZI Files**
    if file_path.endswith(".czi"):
        if not CZIFILE_AVAILABLE:
            print("Error: czifile module not installed. Cannot process .czi files.")
            return None, None

        # Load .czi file
        #czi = czifile.CziFile(file_path)
        image_stack = czifile.imread(file_path)

        # Extract the first frame (T=0) for two channels
        frame_0 = image_stack[0, :, 0, :, :, 0]  # Assuming (T, Z, C, Y, X, S)
        green_channel = frame_0[0, :, :]
        red_channel = frame_0[1, :, :]

    # **Handle TIFF Files**
    elif file_path.endswith(".tif") or file_path.endswith(".tiff"):
        image_stack = tiff.imread(file_path)

        # **Print shape for debugging**
        # print("Loaded TIFF shape:", image.shape)

        # **Handle TIFF stacks (4D format like yours)**
        if len(image_stack.shape) == 4 and image_stack.shape[1] == 2:
            print("Detected TIFF stack with multiple frames. Using first frame (T=0).")
            image_stack = image_stack[0, :, :, :]  # Extract the first frame

        # **Handle standard 3D TIFF (C, Y, X)**
        if len(image_stack.shape) == 3 and image_stack.shape[0] == 2:
            green_channel = image_stack[0, :, :]
            red_channel = image_stack[1, :, :]

        # **Handle alternative format (Y, X, C)**
        elif len(image_stack.shape) == 3 and image_stack.shape[-1] == 2:
            green_channel = image_stack[:, :, 0]
            red_channel = image_stack[:, :, 1]

        else:
            print("Error: TIFF file does not contain exactly 2 channels.")
            return None, None

    else:
        print("Unsupported file format. Please select a .czi or .tif file.")
        return None, None

    # Create colormaps for visualization
    green_cmap = LinearSegmentedColormap.from_list('green_cmap', ['white', 'green'], N=256)
    red_cmap = LinearSegmentedColormap.from_list('red_cmap', ['white', 'red'], N=256)

    # Display individual channels
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(green_channel, cmap=green_cmap)
    plt.title('Green Channel')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(red_channel, cmap=red_cmap)
    plt.title('Red Channel')
    plt.axis('off')

    plt.show()

    # Create overlay image
    overlay = np.zeros((green_channel.shape[0], green_channel.shape[1], 3))
    overlay[:, :, 0] = green_channel / np.max(green_channel)  # Green in the G channel
    overlay[:, :, 1] = red_channel / np.max(red_channel)  # Red in the R channel

    # Display overlay
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title('Overlay of Green and Red Channels')
    plt.axis('off')
    plt.show()

    # Save as multi-channel TIFF
    output_file_preregistration = os.path.join(output_dir, 'preregistration_image.tiff')
    multi_channel_image = np.stack((green_channel, red_channel), axis=0)  # [C, Y, X]
    tiff.imwrite(output_file_preregistration, multi_channel_image.astype(np.uint16),
                 photometric='minisblack', planarconfig='separate', metadata={'axes': 'CYX'})

    print(f"Multi-channel TIFF saved at {output_file_preregistration}")

    return file_path, output_dir
