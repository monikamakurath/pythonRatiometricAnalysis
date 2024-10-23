# Monika A. Makurath
# Step-by-step segmentation: Load, sum intensities for segmentation, and save the summed image
# libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

#%% Load multi-channel registered image stack from specified path
stack_path = '/Users/monikamakurath/Documents/pythonTestFiles/registered_first_frame_16bit.tiff'  # Specify the path to the registered image stack

# Load the multi-channel stack (assuming two channels: green and red)
multi_channel_image = tiff.imread(stack_path)

# Extract the channels
green_channel = multi_channel_image[1, :, :].astype(np.uint16)  # Second channel is green
registered_red_channel = multi_channel_image[0, :, :].astype(np.uint16)  # First channel is red (far-red)

#%% Create a composite image by summing the intensities of the two channels
summed_image = green_channel + registered_red_channel

#%% Display the Green, Red, and Summed Channels side by side
plt.figure(figsize=(18, 6))

# Display the Green Channel
plt.subplot(1, 3, 1)
plt.imshow(green_channel, cmap='gray')
plt.title('Green Channel')
plt.colorbar()

# Display the Red Channel
plt.subplot(1, 3, 2)
plt.imshow(registered_red_channel, cmap='gray')
plt.title('Red (Far-Red) Channel')
plt.colorbar()

# Display the Summed Image
plt.subplot(1, 3, 3)
plt.imshow(summed_image, cmap='gray')
plt.title('Summed Image (Green + Red)')
plt.colorbar()

plt.show()

#%% Save the Summed Image as a TIFF file
output_file_summed = '/Users/monikamakurath/Documents/pythonTestFiles/summed_image.tiff'
tiff.imwrite(output_file_summed, summed_image.astype(np.uint16), photometric='minisblack')

print(f"Summed image saved at {output_file_summed}")
