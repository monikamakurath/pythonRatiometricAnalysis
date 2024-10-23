# Monika A. Makurath
# Perform ratiometric analysis without normalization, display results, and plot ratiometric intensity over time
# libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from matplotlib import cm  # For applying colormap to save RGB image

#%% Load the registered channels and the mask
# PC
stack_path = '/Users/makurathm/Documents/pythonTestFiles/registered_first_frame_16bit.tiff'  # Registered image stack path
mask_path = '/Users/makurathm/Documents/pythonTestFiles/cleaned_mask.tiff'  # Mask path
# laptop
#stack_path = '/Users/monikamakurath/Documents/pythonTestFiles/registered_first_frame_16bit.tiff'  # Registered image stack path
#mask_path = '/Users/monikamakurath/Documents/pythonTestFiles/cleaned_mask.tiff'  # Mask path

multi_channel_image = tiff.imread(stack_path)
cleaned_mask = tiff.imread(mask_path)

# Extract the green and red channels
green_channel = multi_channel_image[1, :, :].astype(np.uint16)  # Green channel
red_channel = multi_channel_image[0, :, :].astype(np.uint16)  # Red channel

#%% Apply the mask to each channel (skip normalization)
masked_green = green_channel * cleaned_mask
masked_red = red_channel * cleaned_mask

#%% Step 1: Compute Ratiometric Intensity (Green / Red) without normalization
epsilon = 1e-10  # Small value to avoid division by zero
ratiometric_intensity_total = np.sum(masked_green) / (np.sum(masked_red) + epsilon)

#%% Step 2: Plot Ratiometric Intensity vs. Time (using frame numbers as time)
frame_rate = 1  # Example: 1 frame per second
num_frames = 1  # For now, we assume 1 frame; adjust this for multiple frames

# Time points for each frame
time_points = np.arange(0, num_frames * frame_rate, frame_rate)

# Plot the ratiometric intensity (total intensity in green / total intensity in red) for the available frames
plt.figure(figsize=(10, 6))
plt.scatter(time_points, [ratiometric_intensity_total] * num_frames, color='b')  # Scatter plot for each frame
plt.title('Ratiometric Intensity vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Ratiometric Intensity (Green / Red)')
plt.grid(True)

# Save the plot as a TIFF file
output_plot_path = '/Users/makurathm/Documents/pythonTestFiles/ratiometric_intensity_vs_time.tiff'
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"Ratiometric intensity plot saved at {output_plot_path}")

#%% Step 3: Perform pixel-wise Ratiometric Analysis for visualization
ratiometric_image = masked_green / (masked_red + epsilon)

#%% Step 4: Check distribution of ratiometric values before display
plt.figure(figsize=(8, 6))
plt.hist(ratiometric_image[ratiometric_image > 0].ravel(), bins=256, color='gray', alpha=0.75)
plt.title('Ratiometric Image Value Distribution')
plt.xlabel('Ratiometric Value')
plt.ylabel('Frequency')
plt.show()

#%% Step 5: Clip extreme values to improve contrast for visualization (optional)
ratiometric_image_clipped = np.clip(ratiometric_image, 0, 5)  # Clipping to improve visualization (adjust as needed)

#%% Step 6: Display the ratiometric image using an appropriate colormap (small values blue, large values red)
plt.figure(figsize=(8, 6))
ratiometric_img_display = plt.imshow(ratiometric_image_clipped, cmap='jet')
plt.colorbar(ratiometric_img_display, label='Ratiometric Value (Green / Red)')
plt.title('Ratiometric Image (Green / Red)')
plt.axis('off')
plt.show()

#%% Step 7: Save the ratiometric image as a TIFF file
output_file_ratiometric = '/Users/makurathm/Documents/pythonTestFiles/ratiometric_image.tiff'
tiff.imwrite(output_file_ratiometric, (ratiometric_image_clipped * 65535).astype(np.uint16))  # Scale back to 16-bit range
print(f"Ratiometric image saved at {output_file_ratiometric}")

#%% Step 8: Save the ratiometric image as an RGB file (colormap applied)
# Normalize ratiometric image to [0, 1] range
ratiometric_image_normalized = (ratiometric_image_clipped - np.min(ratiometric_image_clipped)) / (
        np.max(ratiometric_image_clipped) - np.min(ratiometric_image_clipped) + epsilon)

# Apply the 'jet' colormap to create the RGB image
ratiometric_image_rgb = cm.jet(ratiometric_image_normalized)

# Remove the alpha channel (4th channel) from the RGB image
ratiometric_image_rgb = (ratiometric_image_rgb[:, :, :3] * 255).astype(np.uint8)  # Convert to 8-bit RGB image

# Save the RGB image as a high-resolution PNG with DPI setting
output_file_ratiometric_rgb = '/Users/makurathm/Documents/pythonTestFiles/ratiometric_image_rgb_highres.png'

# Create a figure with the desired DPI and save
fig, ax = plt.subplots(figsize=(ratiometric_image_rgb.shape[1] / 100, ratiometric_image_rgb.shape[0] / 100), dpi=300)
ax.imshow(ratiometric_image_rgb)
ax.axis('off')  # Remove axis for cleaner image

# Save the high-resolution PNG
fig.savefig(output_file_ratiometric_rgb, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)

print(f"Ratiometric image saved as high-resolution RGB PNG at {output_file_ratiometric_rgb}")

