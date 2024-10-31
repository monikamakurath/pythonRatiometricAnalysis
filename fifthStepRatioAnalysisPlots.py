# fifth_step_ratiometric_analysis.py

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import pandas as pd
import seaborn as sns
import os

# Set seaborn styling
sns.set(style='whitegrid', context='talk', font_scale=1.2)


def fifthStepRatioAnalysisPlots(output_dir, frame_rate):
    # Define paths for the registered stack, mask, and output files
    stack_path = os.path.join(output_dir, 'registered_stack_16bit.tiff')
    mask_path = os.path.join(output_dir, 'segmentation_masks_stack.tiff')

    # Output save paths
    output_file_plot_raw_int_svg = os.path.join(output_dir, 'total_pixel_intensity_plot.svg')
    output_file_plot_ratio_int_svg = os.path.join(output_dir, 'ratio_intensity_plot.svg')
    output_file_plot_ratio_int_combined_svg = os.path.join(output_dir, 'ratio_intensity_combined_plot.svg')
    output_excel_path = os.path.join(output_dir, 'total_intensity_data.xlsx')

    # Load the registered image stack and mask
    image_stack = tiff.imread(stack_path)
    mask_stack = tiff.imread(mask_path)

    # Split the two channels into two separate stacks and apply the mask
    green_channel_stack = image_stack[:, 0, :, :] * mask_stack  # Green channel
    red_channel_stack = image_stack[:, 1, :, :] * mask_stack  # Red channel

    # Calculate total pixel intensity for each frame in both stacks
    green_intensity = np.sum(green_channel_stack, axis=(1, 2))
    red_intensity = np.sum(red_channel_stack, axis=(1, 2))
    ratio_intensity = green_intensity / red_intensity
    ratio_baseline_corrected = ratio_intensity - ratio_intensity[0] + 1  # Adjust baseline to 1

    # Create time array in minutes
    num_frames = green_channel_stack.shape[0]
    frames = np.arange(num_frames) * frame_rate / 60  # Convert time to minutes

    # Plot and save intensity and ratio graphs
    plot_and_save(frames, green_intensity, red_intensity, ratio_baseline_corrected,
                  output_file_plot_raw_int_svg, output_file_plot_ratio_int_svg,
                  output_file_plot_ratio_int_combined_svg)

    # Save data to Excel
    save_to_excel(frames, green_intensity, red_intensity, ratio_intensity, output_excel_path)
    print(f"Plots saved and data written to {output_excel_path}")


def plot_and_save(frames, green_intensity, red_intensity, ratio_intensity,
                  raw_int_svg, ratio_svg, combined_svg):
    # Plot and save raw intensities
    plt.figure(figsize=(10, 6))
    plt.plot(frames, green_intensity, label='Green Channel (Total Intensity)', color='green', linewidth=2)
    plt.plot(frames, red_intensity, label='Red Channel (Total Intensity)', color='red', linewidth=2)
    plt.xlabel('Time (min)')
    plt.ylabel('Total Pixel Intensity')
    plt.title('Total Pixel Intensity (Green and Red Channels)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(raw_int_svg, format='svg', dpi=300)

    # Plot and save intensity ratio
    plt.figure(figsize=(10, 6))
    plt.plot(frames, ratio_intensity, color='black', linewidth=2)
    plt.xlabel('Time (min)')
    plt.ylabel('Intensity Ratio (Green/Red)')
    plt.title('Green/Red Intensity Ratio')
    plt.ylim(0, 2)
    plt.tight_layout()
    plt.savefig(ratio_svg, format='svg', dpi=300)

    # Combined plot with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 16))
    ax1.plot(frames, green_intensity, label='Green Channel (Total Intensity)', color='green', linewidth=2)
    ax1.plot(frames, red_intensity, label='Red Channel (Total Intensity)', color='red', linewidth=2)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Total Pixel Intensity')
    ax1.legend(frameon=False)
    ax2.plot(frames, ratio_intensity, color='black', linewidth=2)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Intensity Ratio (Green/Red)')
    ax2.set_ylim(0, 2)
    plt.tight_layout()
    plt.savefig(combined_svg, format='svg', dpi=300)


def save_to_excel(frames, green_intensity, red_intensity, ratio_intensity, output_excel_path):
    # Save data to an Excel spreadsheet
    data = {
        'Time (s)': frames * 60,  # Convert back to seconds
        'Green Intensity': green_intensity,
        'Red Intensity': red_intensity,
        'Green/Red Ratio': ratio_intensity
    }
    df = pd.DataFrame(data)
    df.to_excel(output_excel_path, index=False)

