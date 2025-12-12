# 12/11/2025 Monika A. Makurath
# However you pre-process the file, make sure to arrange your channels
# such that the first channel gets divided by the second.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
import os
import math
from matplotlib import cm
import cmocean
import getpass
from tkinter import Tk, filedialog  # <-- added for file chooser

plt.rcParams['font.family'] = 'Arial'  # Set font for all images


def main():
    # 1. Let the user pick a TIFF file
    tiff_path = choose_tiff_file()
    if not tiff_path:
        print("No file selected. Exiting.")
        return

    print(f"Selected file: {tiff_path}")

    # 2. Load the TIFF file
    image = tiff.imread(tiff_path)

    print(f"Loaded image with shape: {image.shape}, dtype: {image.dtype}")

    # 3. Interpret it as two channels and extract them
    # We handle common layouts: (2, Y, X) or (Y, X, 2)
    if image.ndim == 3 and image.shape[0] == 2:
        # Shape: (C, Y, X)
        ch1 = image[0].astype(np.float32)
        ch2 = image[1].astype(np.float32)
    elif image.ndim == 3 and image.shape[-1] == 2:
        # Shape: (Y, X, C)
        ch1 = image[..., 0].astype(np.float32)
        ch2 = image[..., 1].astype(np.float32)
    else:
        raise ValueError(
            f"Expected a 2-channel image with shape (2, Y, X) or (Y, X, 2), "
            f"but got {image.shape}"
        )


    # Normalize to scale by first-frame intensity, to have time-normalized ratio
    ch1 = ch1 / np.sum(ch1[0] + 1e-10)
    ch2 = ch2 / np.sum(ch2[0] + 1e-10)

    ratio = compute_ratiometric_stack(ch1, ch2)

    # Replace zeros with NaN's so that the background is treated differently than the signal when the color bar is applied
    ratio[ratio == 0] = np.nan



    # 5. Display the ratio image
    plt.subplots(figsize=(8, 8))
    im = plt.imshow(np.clip(ratio, 0.5, 1.5), cmap=cmocean.cm.matter)
    plt.title("Ratiometric image (channel 1 / channel 2)")
    plt.axis("off")
    plt.colorbar(im, label="Ratio")

    # ---- SAVE HIGH-RES TIFF ----
    directory, filename = os.path.split(tiff_path)
    basename, ext = os.path.splitext(filename)
    save_path = os.path.join(directory, f"{basename}_ratio.tiff")

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Saved high-resolution ratio image to: {save_path}")

    plt.show()


def compute_ratiometric_stack(green_stack, red_stack):
    epsilon = 1e-10  # Small constant to avoid division by zero
    ratiometric_stack = np.zeros_like(green_stack, dtype=np.float32)

    for i in range(green_stack.shape[0]):
        ratiometric_frame = green_stack[i] / (red_stack[i] + epsilon)
        ratiometric_stack[i] = ratiometric_frame
    return ratiometric_stack

def choose_tiff_file():
    """Open a dialog to choose a TIFF file and return its path."""
    root = Tk()
    root.withdraw()  # hide the main Tk window
    file_path = filedialog.askopenfilename(
        title="Select a 2-channel TIFF file",
        filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path



if __name__ == "__main__":
    main()
