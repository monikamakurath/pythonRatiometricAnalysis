# Monika A. Makurath
# Apply mean threshold, remove small speckles, mask the channels, and save side-by-side image
# libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.morphology import remove_small_objects
from matplotlib.colors import LinearSegmentedColormap

#%% Load the previously saved summed image and registered channels
summed_image_path = '/Users/monikamakurath/Documents/pythonTestFiles/summed_image.tiff'
stack_path = '/Users/monikamakurath/Documents/pythonTestFiles/registered_first_frame_16bit.tiff'  # Registered image stack path

summed_image = tiff.imread(summed_image_path)
multi_channel_image = tiff.imread(stack_path)

# Extract the green and red channels
green_channel = multi_channel_image[1, :, :].astype(np.uint16)  # Green channel
red_channel = multi_channel_image[0, :, :].astype(np.uint16)  # Red channel

#%% Step 1: Apply Mean Threshold to create a binary mask
mean_threshold_value = np.mean(summed_image)  # Calculate mean intensity
binary_mask = summed_image > mean_threshold_value  # Create binary mask

#%% Step 2: Remove small speckles and background particles
min_size = 500  # Adjust based on the size of your cells
cleaned_mask = remove_small_objects(binary_mask, min_size=min_size)

#%% Save the cleaned mask as a TIFF file
output_file_mask = '/Users/monikamakurath/Documents/pythonTestFiles/cleaned_mask.tiff'
tiff.imwrite(output_file_mask, cleaned_mask.astype(np.uint16), photometric='minisblack')
print(f"Cleaned mask saved at {output_file_mask}")

#%% Step 3: Apply the mask to each channel
masked_green = green_channel * cleaned_mask
masked_red = red_channel * cleaned_mask

#%% Step 4: Display the cleaned mask, masked green, and red channels
# Create custom colormaps for green and red on black background
green_cmap = LinearSegmentedColormap.from_list('green_cmap', ['black', 'green'], N=256)
red_cmap = LinearSegmentedColormap.from_list('red_cmap', ['black', 'red'], N=256)

plt.figure(figsize=(18, 6))

# Display the Mask
plt.subplot(1, 3, 1)
plt.imshow(cleaned_mask, cmap='gray')
plt.title('Cleaned Mask')
plt.axis('off')

# Display the Masked Green Channel
plt.subplot(1, 3, 2)
plt.imshow(masked_green, cmap=green_cmap)
plt.title('Masked Green Channel')
plt.axis('off')

# Display the Masked Red Channel
plt.subplot(1, 3, 3)
plt.imshow(masked_red, cmap=red_cmap)
plt.title('Masked Red Channel')
plt.axis('off')

plt.show()

#%% Step 5: Save the masked green and red channels side by side as a single image
# Create an empty array to hold both channels side by side
height, width = masked_green.shape
side_by_side_image = np.zeros((height, 2 * width), dtype=np.uint16)

# Place green channel on the left and red channel on the right
side_by_side_image[:, :width] = masked_green  # Green on the left
side_by_side_image[:, width:] = masked_red    # Red on the right

# Save the side-by-side image as a TIFF
output_file_side_by_side = '/Users/monikamakurath/Documents/pythonTestFiles/masked_green_red_side_by_side.tiff'
tiff.imwrite(output_file_side_by_side, side_by_side_image)

print(f"Masked green and red channels saved side by side at {output_file_side_by_side}")
