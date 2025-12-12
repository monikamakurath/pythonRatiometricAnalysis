import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import pandas as pd
import seaborn as sns
import os

# Set seaborn styling
sns.set(style='whitegrid', context='talk', font_scale=1.2)


def ratioAnalysisPlots(output_dir, frame_rate):
    # Define file paths
    stack_path = os.path.join(output_dir, 'registered_stack_16bit.tiff')
    mask_path = os.path.join(output_dir, 'segmentation_masks_stack.tiff')
    output_excel_path = os.path.join(output_dir, 'total_intensity_data.xlsx')

    # Output paths for plots
    output_plots = {
        "norm_and_ratio_intensity": os.path.join(output_dir, "normalized_and_ratio_intensity_plot.svg")
    }

    # Load the registered image stack and mask
    image_stack = tiff.imread(stack_path)
    mask_stack = tiff.imread(mask_path)

    # Separate channels and apply mask
    green_channel_stack = image_stack[:, 0, :, :] * mask_stack
    red_channel_stack = image_stack[:, 1, :, :] * mask_stack

    # Compute total intensity for each frame
    raw_green_intensity = np.sum(green_channel_stack, axis=(1, 2))
    raw_red_intensity = np.sum(red_channel_stack, axis=(1, 2))

    # Normalize intensities by first frame
    norm_green_intensity = raw_green_intensity / (raw_green_intensity[0] + 1e-10)
    norm_red_intensity = raw_red_intensity / (raw_red_intensity[0] + 1e-10)

    # Take the ratio
    ratio_intensity = norm_green_intensity / norm_red_intensity

    # Compute time array (in minutes)
    time_array = np.arange(green_channel_stack.shape[0]) * frame_rate / 60

    # Generate and save plots
    plot_and_save(time_array, norm_green_intensity, norm_red_intensity, ratio_intensity, output_plots)

    # Save data to Excel
    save_to_excel(time_array, raw_green_intensity, raw_red_intensity, norm_green_intensity, norm_red_intensity, ratio_intensity, output_excel_path)

    print(f"Plots saved and data written to {output_excel_path}")


def plot_and_save(time_array, norm_green_intensity, norm_red_intensity, ratio_intensity, output_plots):
    """Creates and saves plots for intensity and ratio analysis."""

    high_dpi = 300

    # Plot total pixel intensities
    plt.figure(figsize=(6, 4), dpi=high_dpi)
    plt.plot(time_array, norm_green_intensity, marker='o', linestyle='-', color='green', markersize=4, linewidth=1,
             label="Normalized Green Channel")
    plt.plot(time_array, norm_red_intensity, marker='s', linestyle='-', color='red', markersize=4, linewidth=1,
             label="Normalized Red Channel")
    plt.plot(time_array, ratio_intensity, marker='s', linestyle='-', color='black', markersize=4, linewidth=1,
             label="Ratio")
    plt.xlabel("Time (min)", fontsize=12, fontweight='bold')
    plt.ylabel("Normalized Intensity", fontsize=12, fontweight='bold')
    plt.ylim(0, max(norm_green_intensity.max(), norm_red_intensity.max(), ratio_intensity.max()) * 1.1)
    plt.legend(fontsize=10, loc="upper right", frameon=False)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_plots["norm_and_ratio_intensity"], format="svg", bbox_inches="tight")
    plt.show()

    print(f"Plots saved: {output_plots}")


def save_to_excel(time_array, raw_green_intensity, raw_red_intensity, norm_green_intensity, norm_red_intensity, ratio_intensity, output_excel_path):
    """Saves intensity data to an Excel file."""
    data = {
        'Time (s)': time_array * 60,  # Convert back to seconds
        'Raw Green Intensity': raw_green_intensity,
        'Raw Red Intensity': raw_red_intensity,
        'Normalized Green Intensity': norm_green_intensity,
        'Normalized Red Intensity': norm_red_intensity,
        'Green/Red Ratio': ratio_intensity
    }
    df = pd.DataFrame(data)
    df.to_excel(output_excel_path, index=False)