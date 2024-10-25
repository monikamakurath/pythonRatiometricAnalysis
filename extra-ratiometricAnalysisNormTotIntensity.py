# Monika A. Makurath
# Normalize pixel intensities, perform ratiometric analysis, display results, and plot ratiometric intensity over time
# libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.morphology import remove_small_objects

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

#%% Apply the mask to each channel
masked_green = green_channel * cleaned_mask
masked_red = red_channel * cleaned_mask

#%% Function to normalize pixel intensities so that the total intensity in each masked region is 1
def normalize_image_to_total_intensity(image):
    total_intensity = np.sum(image[image > 0])  # Sum of intensities within the masked region (non-zero pixels)
    normalized_image = image / (total_intensity + 1e-10)  # Normalize so the total sum equals 1
    return normalized_image

#%% Step 1: Normalize the masked green and red channels based on total intensity
normalized_green = normalize_image_to_total_intensity(masked_green)
normalized_red = normalize_image_to_total_intensity(masked_red)

# Check the total intensity after normalization
total_intensity_green_after = np.sum(normalized_green)
total_intensity_red_after = np.sum(normalized_red)
print(f"Total Green Intensity after Normalization: {total_intensity_green_after}")
print(f"Total Red Intensity after Normalization: {total_intensity_red_after}")

#%% Step 2: Compute Ratiometric Intensity (Green / Red) for each frame
# Here we'll calculate the ratiometric intensity as the total intensity in the green channel
# divided by the total intensity in the red channel within the masked region
ratiometric_intensity_total = np.sum(normalized_green) / np.sum(normalized_red)

#%% Step 3: Plot Ratiometric Intensity vs. Time (using frame numbers as time)
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
# PC
output_plot_path = '/Users/makurathm/Documents/pythonTestFiles/ratiometric_intensity_vs_time.tiff'
# laptop
#output_plot_path = '/Users/monikamakurath/Documents/pythonTestFiles/ratiometric_intensity_vs_time.tiff'
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"Ratiometric intensity plot saved at {output_plot_path}")

#%% Step 4: Perform Ratiometric Analysis for visualization
epsilon = 1e-10
ratiometric_image = normalized_green / (normalized_red + epsilon)

#%% Step 5: Check distribution of ratiometric values before display
plt.figure(figsize=(8, 6))
plt.hist(ratiometric_image[ratiometric_image > 0].ravel(), bins=256, color='gray', alpha=0.75)
plt.title('Ratiometric Image Value Distribution')
plt.xlabel('Ratiometric Value')
plt.ylabel('Frequency')
plt.show()

#%% Step 6: Clip extreme values to improve contrast for visualization (optional)
ratiometric_image_clipped = np.clip(ratiometric_image, 0, 5)  # Clipping to improve visualization (adjust as needed)

#%% Step 7: Display the ratiometric image using an appropriate colormap (small values blue, large values red)
plt.figure(figsize=(8, 6))
ratiometric_img_display = plt.imshow(ratiometric_image_clipped, cmap='jet')
plt.colorbar(ratiometric_img_display, label='Ratiometric Value (Green / Red)')
plt.title('Ratiometric Image (Green / Red)')
plt.axis('off')
plt.show()

#%% Step 8: Save the ratiometric image as a TIFF file
# PC
output_file_ratiometric = '/Users/makurathm/Documents/pythonTestFiles/ratiometric_image.tiff'
# laptop
#output_file_ratiometric = '/Users/monikamakurath/Documents/pythonTestFiles/ratiometric_image.tiff'

# Save the ratiometric image
tiff.imwrite(output_file_ratiometric, (ratiometric_image_clipped * 65535).astype(np.uint16))  # Scale back to 16-bit range

print(f"Ratiometric image saved at {output_file_ratiometric}")
