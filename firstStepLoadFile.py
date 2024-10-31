# firstStepLoadFile.py

import czifile
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import tifffile as tiff
import os
from tkinter import Tk, filedialog

def firstStepLoadFile():
    # Prompt user to select a .czi file
    Tk().withdraw()  # Hide main tkinter window
    file_path = filedialog.askopenfilename(title="Select .czi File", filetypes=[("CZI files", "*.czi")])
    if not file_path:
        print("No file selected.")
        return

    # Generate output directory based on the input file name
    file_name = os.path.basename(file_path)
    output_dir = os.path.join(os.path.dirname(file_path), f"{file_name}_output")
    os.makedirs(output_dir, exist_ok=True)

    # Load .czi file and image data
    czi = czifile.CziFile(file_path)
    image = czifile.imread(file_path)

    # Initialize path for pre-registration image
    output_file_preregistration = os.path.join(output_dir, 'preregistration_image.tiff')

    # Extract the first frame (T=0) for two channels
    frame_0 = image[0, :, 0, :, :, 0]
    green_channel = frame_0[0, :, :]  # Green channel
    red_channel = frame_0[1, :, :]    # Red channel

    # Create colormaps
    green_cmap = LinearSegmentedColormap.from_list('green_cmap', ['white', 'green'], N=256)
    red_cmap = LinearSegmentedColormap.from_list('red_cmap', ['white', 'red'], N=256)

    # Display channels
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

    # Overlay image for pre-registration record
    overlay = np.zeros((green_channel.shape[0], green_channel.shape[1], 3))
    overlay[:, :, 1] = green_channel / np.max(green_channel)
    overlay[:, :, 0] = red_channel / np.max(red_channel)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title('Overlay of Green and Red Channels')
    plt.axis('off')
    plt.show()

    # Save as multi-channel TIFF
    multi_channel_image = np.stack((red_channel, green_channel), axis=0)
    tiff.imwrite(output_file_preregistration, multi_channel_image.astype(np.uint16), photometric='minisblack', planarconfig='separate', metadata={'axes': 'CYX'})

    # return the file path
    return file_path, output_dir

    print(f"Multi-channel TIFF saved at {output_file_preregistration}")
