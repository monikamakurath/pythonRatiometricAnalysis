# sixth_step_ratiometric_image_creation.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
import os
import math
from matplotlib import cm
import cmocean
import getpass

plt.rcParams['font.family'] = 'Arial'  # Set font for all images

def ratioAnalysisImageStack(output_dir, frame_rate, pixel_width, scale_bar_length_microns, high_dpi):
    # Paths for the registered stack, mask, and output files
    stack_path = os.path.join(output_dir, 'registered_stack_16bit.tiff')
    mask_path = os.path.join(output_dir, 'segmentation_masks_stack.tiff')
    output_file_rgb_stack = os.path.join(output_dir, 'ratiometric_stack.tiff')
    output_file_last_frame = os.path.join(output_dir, 'last_frame_high_res.tiff')
    color_bar_horizontal_path = os.path.join(output_dir, 'color_bar_horizontal.svg')
    color_bar_vertical_path = os.path.join(output_dir, 'color_bar_vertical.svg')

    # Load the registered image stack and mask
    image_stack = tiff.imread(stack_path)
    mask_stack = tiff.imread(mask_path)

    # Separate channels and apply mask
    green_stack = image_stack[:, 0, :, :] * mask_stack  # Green channel
    red_stack = image_stack[:, 1, :, :] * mask_stack    # Red channel

    # Normalize to scale by first-frame intensity, to have time-normalized ratio
    green_stack = green_stack/np.sum(green_stack[0] + 1e-10)
    red_stack = red_stack / np.sum(red_stack[0] + 1e-10)

    # Ratio
    ratiometric_stack = compute_ratiometric_stack(green_stack, red_stack)

    # Replace zeros with NaN's so that the background is treated differently than the signal when the color bar is applied
    ratiometric_stack[ratiometric_stack == 0] = np.nan

    # Compute time array in minutes
    num_frames = len(ratiometric_stack)
    time_array = np.arange(num_frames) * frame_rate / 60.0

    # Define scale bar length in pixels
    scale_bar_length_pixels = scale_bar_length_microns / pixel_width
    formatted_frames = format_frames(ratiometric_stack, time_array, scale_bar_length_pixels, high_dpi)

    # Save all frames to a single TIFF stack and the color bars
    tiff.imwrite(output_file_rgb_stack, np.array(formatted_frames), photometric='rgb')
    save_color_bars(color_bar_horizontal_path, color_bar_vertical_path, high_dpi)
    print(f"Ratiometric stack saved to {output_file_rgb_stack}. Color bars saved as SVGs.")

def compute_ratiometric_stack(green_stack, red_stack):
    epsilon = 1e-10  # Small constant to avoid division by zero
    ratiometric_stack = np.zeros_like(green_stack, dtype=np.float32)

    for i in range(green_stack.shape[0]):
        ratiometric_frame = green_stack[i] / (red_stack[i] + epsilon)
        ratiometric_stack[i] = ratiometric_frame
    return ratiometric_stack

def format_frames(ratiometric_stack, time_array, scale_bar_length_pixels, high_dpi):
    formatted_frames = []

    for i, frame in enumerate(ratiometric_stack):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=high_dpi)
        #ax.imshow(np.clip(frame, 0, 3), cmap='coolwarm')
        #ax.imshow(np.clip(frame, 0, 2), cmap='coolwarm') # "RdGy" "coolwarm" "binary" "turbo"
        ax.imshow(np.clip(frame, 0.2, 1.7), cmap=cmocean.cm.matter) # deep matter
        ax.axis('off')

        # Add time text
        ax.text(0.02, 0.95, f'{math.ceil(time_array[i])} min', transform=ax.transAxes, fontsize=18, color='black', verticalalignment='top')

        # Add scale bar
        scale_bar = patches.Rectangle((frame.shape[1] - scale_bar_length_pixels - 10, frame.shape[0] - 10), scale_bar_length_pixels, 2, edgecolor='black', facecolor='black')
        ax.add_patch(scale_bar)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        rgba_buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba_frame = rgba_buffer.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        formatted_frames.append(rgba_frame[:, :, :3])
        plt.close(fig)
    return formatted_frames

def save_color_bars(horizontal_path, vertical_path, high_dpi):
    # Horizontal color bar
    fig, ax = plt.subplots(figsize=(6, 1), dpi=high_dpi)
    plt.colorbar(cm.ScalarMappable(cmap=cmocean.cm.matter), cax=ax, orientation='horizontal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(horizontal_path, dpi=high_dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Vertical color bar
    fig, ax = plt.subplots(figsize=(1, 6), dpi=high_dpi)
    plt.colorbar(cm.ScalarMappable(cmap=cmocean.cm.matter), cax=ax, orientation='vertical')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(vertical_path, dpi=high_dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

