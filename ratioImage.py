# 12/11/2025 Monika A. Makurath
# However you pre-process the file, make sure to arrange your channels
# such that the first channel gets divided by the second.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
import os
import math
from matplotlib import cm, colors
import cmocean
import getpass
from tkinter import Tk, filedialog  # <-- added for file chooser

plt.rcParams['font.family'] = 'Arial'  # Set font for all images

# Try importing czifile (only needed for .czi files)
try:
    import czifile
    CZIFILE_AVAILABLE = True
except ImportError:
    CZIFILE_AVAILABLE = False
    print("czifile not found, .czi loading will not be available.")


def main():
    # 1. Let the user pick a file (.tif/.tiff/.czi)
    img_path = choose_image_file()
    if not img_path:
        print("No file selected. Exiting.")
        return

    print(f"Selected file: {img_path}")

    # 2. Load the image (TIFF or CZI)
    ext = os.path.splitext(img_path)[1].lower()

    if ext == ".czi":
        if not CZIFILE_AVAILABLE:
            raise ImportError("czifile is not installed, cannot load .czi files.")
        # czifile.imread returns a NumPy array; often with extra singleton dims
        image = czifile.imread(img_path)
        image = np.squeeze(image)  # drop singleton axes; we'll interpret below
        print(f"Loaded CZI image with shape: {image.shape}, dtype: {image.dtype}")
    else:
        # Default: load as TIFF
        image = tiff.imread(img_path)
        print(f"Loaded TIFF image with shape: {image.shape}, dtype: {image.dtype}")

    # 3. Interpret as two channels (supports single frame and time stack)
    # Expect shapes like:
    #   (T, 2, Y, X)
    #   (2, Y, X)
    #   (Y, X, 2)
    if image.ndim == 4 and image.shape[1] == 2:
        # Shape: (T, C, Y, X)
        ch1 = image[:, 0].astype(np.float32)   # (T, Y, X)
        ch2 = image[:, 1].astype(np.float32)   # (T, Y, X)
        is_stack = True

    elif image.ndim == 3 and image.shape[0] == 2:
        # Shape: (C, Y, X) — single timepoint
        ch1 = image[0].astype(np.float32)[None, ...]  # add T axis -> (1, Y, X)
        ch2 = image[1].astype(np.float32)[None, ...]
        is_stack = False

    elif image.ndim == 3 and image.shape[-1] == 2:
        # Shape: (Y, X, C) — single timepoint
        ch1 = image[..., 0].astype(np.float32)[None, ...]
        ch2 = image[..., 1].astype(np.float32)[None, ...]
        is_stack = False

    else:
        raise ValueError(
            f"Expected 2-channel data with shape (T, 2, Y, X), (2, Y, X) or (Y, X, 2), "
            f"but got {image.shape}"
        )

    # 4. Raw ratio, no normalization
    epsilon = 1e-10
    ratio = ch1 / (ch2 + epsilon)  # shape (T, Y, X)

    # ---- BACKGROUND HANDLING ----
    WHITE_BACKGROUND = 0  # 1 = make dim background white, 0 = keep as is
    BG_PERCENTILE = 50    # pixels in lowest X% of (ch1+ch2) are "background"

    if WHITE_BACKGROUND:
        signal = ch1 + ch2  # (T, Y, X)
        thresh = np.nanpercentile(signal, BG_PERCENTILE)
        bg = signal <= thresh  # dim in both channels
        ratio[bg] = np.nan     # will be drawn with cmap.set_bad()
    else:
        bg = np.zeros_like(ratio, dtype=bool)  # no background masking

    print("ratio global min, max, mean:",
          float(np.nanmin(ratio)), float(np.nanmax(ratio)), float(np.nanmean(ratio)))

    # 5. Auto-contrast from data itself
    vmin = np.nanpercentile(ratio, 25)
    vmax = np.nanpercentile(ratio, 75)
    print("display vmin, vmax:", vmin, vmax)

    # Common paths + frame count
    directory, filename = os.path.split(img_path)
    basename, ext = os.path.splitext(filename)
    T = ratio.shape[0]

    # ---- BUILD RGB STACK WITH SAME COLORMAP & LIMITS ----
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cmocean.cm.matter.copy()
    cmap.set_bad(color='white')  # NaNs → white

    formatted_frames = []
    for t in range(T):
        rgba = cmap(norm(ratio[t]))                 # (Y, X, 4), float 0–1
        rgb = (rgba[..., :3] * 255).astype(np.uint8)  # (Y, X, 3), uint8

        # If you want to FORCE white background in the RGB too:
        if WHITE_BACKGROUND:
            rgb[bg[t]] = 255  # set background pixels to white

        formatted_frames.append(rgb)

    stack_rgb_path = os.path.join(directory, f"{basename}_ratio_rgb_stack.tiff")
    tiff.imwrite(stack_rgb_path, np.array(formatted_frames), photometric="rgb")
    print(f"Saved RGB ratio stack to: {stack_rgb_path}")

    # ---- DISPLAY & SAVE ONLY FIRST AND LAST FRAMES ----
    if T == 1:
        frame_indices = [0]
    else:
        frame_indices = [0, T - 1]  # first and last

    for t in frame_indices:
        plt.figure(figsize=(8, 8))
        im = plt.imshow(ratio[t], cmap=cmap, vmin=vmin, vmax=vmax)
        title = (
            f"Raw ratio (ch1 / ch2), frame {t}"
            if T > 1 else "Raw ratio (ch1 / ch2)"
        )
        plt.title(title)
        plt.axis("off")
        plt.colorbar(im, label="Ratio")

        if T == 1:
            panel_path = os.path.join(directory, f"{basename}_ratio.tiff")
        else:
            label = "first" if t == 0 else "last"
            panel_path = os.path.join(directory, f"{basename}_ratio_{label}.tiff")

        plt.savefig(panel_path, dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Saved high-resolution ratio panel to: {panel_path}")

        plt.show()


def choose_image_file():
    """Open a dialog to choose a TIFF or CZI file and return its path."""
    root = Tk()
    root.withdraw()  # hide the main Tk window
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[
            ("CZI files", "*.czi"),
            ("TIFF files", "*.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return file_path


if __name__ == "__main__":
    main()
