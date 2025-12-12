import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import pandas as pd
import seaborn as sns
import os

# Set seaborn styling
sns.set(style='whitegrid', context='talk', font_scale=1.2)


def subcellularRatioAnalysisPlots(output_dir, frame_rate, file_path):
    # Define file paths
    stack_path = os.path.join(output_dir, 'registered_stack_16bit.tiff')
    whole_cell_mask_path = os.path.join(output_dir, 'segmentation_masks_stack.tiff')
    base_name = os.path.splitext(file_path)[0]
    nucleus_mask_path = base_name + "-nucleus-masks.tif"
    periphery_mask_path = base_name + "-periphery-masks.tif"
    output_excel_path = os.path.join(output_dir, 'total_ratio_data.xlsx')
    output_plot_path = os.path.join(output_dir, 'total_ratio_plot.svg')

    # Output paths for plots
    whole_cell_output_plots = {
        "whole_cell_norm_and_ratio_intensity": os.path.join(output_dir, "whole_cell_normalized_and_ratio_intensity_plot.svg")
    }
    nucleus_output_plots = {
        "nucleus_norm_and_ratio_intensity": os.path.join(output_dir,
                                                            "nucleus_normalized_and_ratio_intensity_plot.svg")
    }
    periphery_output_plots = {
        "whole_cell_norm_and_ratio_intensity": os.path.join(output_dir,
                                                            "periphery_normalized_and_ratio_intensity_plot.svg")
    }
    cell_body_output_plots = {
        "whole_cell_norm_and_ratio_intensity": os.path.join(output_dir,
                                                            "cell_body_normalized_and_ratio_intensity_plot.svg")
    }

    # Load the registered image stack and mask
    image_stack = tiff.imread(stack_path)
    whole_cell_mask_stack = tiff.imread(whole_cell_mask_path)
    nucleus_cell_mask_stack = tiff.imread(nucleus_mask_path)
    periphery_cell_mask_stack = tiff.imread(periphery_mask_path)

    # Need to invert the binary mask
    nucleus_cell_mask_stack = (nucleus_cell_mask_stack == 0).astype(np.uint8)
    periphery_cell_mask_stack = (periphery_cell_mask_stack == 0).astype(np.uint8)

    # Create cell body mask
    cell_body_mask_stack = np.logical_and(
        whole_cell_mask_stack,
        np.logical_not(np.logical_or(nucleus_cell_mask_stack, periphery_cell_mask_stack))
    )

    nucleus_cell_mask_stack = np.logical_and(nucleus_cell_mask_stack, whole_cell_mask_stack).astype(np.uint8)
    periphery_cell_mask_stack = np.logical_and(periphery_cell_mask_stack, whole_cell_mask_stack).astype(np.uint8)


    # Separate channels and apply mask
    whole_cell_green_channel_stack = image_stack[:, 0, :, :] * whole_cell_mask_stack
    whole_cell_red_channel_stack = image_stack[:, 1, :, :] * whole_cell_mask_stack
    nucleus_green_channel_stack = image_stack[:, 0, :, :] * nucleus_cell_mask_stack
    nucleus_red_channel_stack = image_stack[:, 1, :, :] * nucleus_cell_mask_stack
    periphery_green_channel_stack = image_stack[:, 0, :, :] * periphery_cell_mask_stack
    periphery_red_channel_stack = image_stack[:, 1, :, :] * periphery_cell_mask_stack
    cell_body_green_channel_stack = image_stack[:, 0, :, :] * cell_body_mask_stack
    cell_body_red_channel_stack = image_stack[:, 1, :, :] * cell_body_mask_stack

    # Compute total intensity for each frame
    whole_cell_raw_green_intensity = np.sum(whole_cell_green_channel_stack, axis=(1, 2))
    whole_cell_raw_red_intensity = np.sum(whole_cell_red_channel_stack, axis=(1, 2))
    nucleus_raw_green_intensity = np.sum(nucleus_green_channel_stack, axis=(1, 2))
    nucleus_raw_red_intensity = np.sum(nucleus_red_channel_stack, axis=(1, 2))
    periphery_raw_green_intensity = np.sum(periphery_green_channel_stack, axis=(1, 2))
    periphery_raw_red_intensity = np.sum(periphery_red_channel_stack, axis=(1, 2))
    cell_body_raw_green_intensity = np.sum(cell_body_green_channel_stack, axis=(1, 2))
    cell_body_raw_red_intensity = np.sum(cell_body_red_channel_stack, axis=(1, 2))

    # Normalize intensities by first frame
    whole_cell_norm_green_intensity = whole_cell_raw_green_intensity / (whole_cell_raw_green_intensity[0] + 1e-10)
    whole_cell_norm_red_intensity = whole_cell_raw_red_intensity / (whole_cell_raw_red_intensity[0] + 1e-10)
    nucleus_norm_green_intensity = nucleus_raw_green_intensity / (nucleus_raw_green_intensity[0] + 1e-10)
    nucleus_norm_red_intensity = nucleus_raw_red_intensity / (nucleus_raw_red_intensity[0] + 1e-10)
    periphery_norm_green_intensity = periphery_raw_green_intensity / (periphery_raw_green_intensity[0] + 1e-10)
    periphery_norm_red_intensity = periphery_raw_red_intensity / (periphery_raw_red_intensity[0] + 1e-10)
    cell_body_norm_green_intensity = cell_body_raw_green_intensity / (cell_body_raw_green_intensity[0] + 1e-10)
    cell_body_norm_red_intensity = cell_body_raw_red_intensity / (cell_body_raw_red_intensity[0] + 1e-10)


    # Take the ratio
    whole_cell_ratio_intensity = whole_cell_norm_green_intensity / whole_cell_norm_red_intensity
    nucleus_ratio_intensity = nucleus_norm_green_intensity / nucleus_norm_red_intensity
    periphery_ratio_intensity = periphery_norm_green_intensity / periphery_norm_red_intensity
    cell_body_ratio_intensity = cell_body_norm_green_intensity / cell_body_norm_red_intensity


    # Compute time array (in minutes)
    time_array = np.arange(whole_cell_green_channel_stack.shape[0]) * frame_rate / 60

    # Generate and save plots
    plot_and_save(time_array, whole_cell_norm_green_intensity, whole_cell_norm_red_intensity,
                  whole_cell_ratio_intensity, whole_cell_output_plots)
    plot_and_save(time_array, nucleus_norm_green_intensity, nucleus_norm_red_intensity,
                  nucleus_ratio_intensity, nucleus_output_plots)
    plot_and_save(time_array, periphery_norm_green_intensity, periphery_norm_red_intensity,
                  periphery_ratio_intensity, periphery_output_plots)
    plot_and_save(time_array, cell_body_norm_green_intensity, cell_body_norm_red_intensity,
                  cell_body_ratio_intensity, cell_body_output_plots)

    # Save data to separate Excel files
    save_to_excel(time_array, whole_cell_raw_green_intensity, whole_cell_raw_red_intensity,
                  whole_cell_norm_green_intensity, whole_cell_norm_red_intensity,
                  whole_cell_ratio_intensity,
                  os.path.join(output_dir, 'whole_cell_ratio_data.xlsx'))

    save_to_excel(time_array, nucleus_raw_green_intensity, nucleus_raw_red_intensity,
                  nucleus_norm_green_intensity, nucleus_norm_red_intensity,
                  nucleus_ratio_intensity,
                  os.path.join(output_dir, 'nucleus_ratio_data.xlsx'))

    save_to_excel(time_array, periphery_raw_green_intensity, periphery_raw_red_intensity,
                  periphery_norm_green_intensity, periphery_norm_red_intensity,
                  periphery_ratio_intensity,
                  os.path.join(output_dir, 'periphery_ratio_data.xlsx'))

    save_to_excel(time_array, cell_body_raw_green_intensity, cell_body_raw_red_intensity,
                  cell_body_norm_green_intensity, cell_body_norm_red_intensity,
                  cell_body_ratio_intensity,
                  os.path.join(output_dir, 'cell_body_ratio_data.xlsx'))

    print(f"Plots saved and data written to {output_excel_path}")


def plot_and_save(time_array, norm_green_intensity, norm_red_intensity, ratio_intensity, output_plots):
    """Creates and saves plots for intensity and ratio analysis."""
    high_dpi = 300

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
    plt.savefig(list(output_plots.values())[0], format="svg", bbox_inches="tight")
    plt.close()
    print(f"✔ Plot saved: {list(output_plots.values())[0]}")


def save_to_excel(time_array, raw_green_intensity, raw_red_intensity,
                  norm_green_intensity, norm_red_intensity, ratio_intensity, output_excel_path):
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
