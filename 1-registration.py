# Monika A. Makurath
# 08-22-24
# first step of fluorescence image analysis
# load file and import essential information

# libraries
import czifile
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import tifffile as tiff
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

#%% load .czi file
file_path = '/Users/monikamakurath/Documents/pythonTestFiles/test.czi'  # Update this path
czi = czifile.CziFile(file_path)
image = czifile.imread(file_path)

#%% show first frame (T=0)
# Extract the first frame (T=0) for two channels
frame_0 = image[0, :, 0, :, :, 0]  # T=0, Z=0, S=0

# Channel 0 = Green, Channel 1 = Red (ensure green is first, red is second)
green_channel = frame_0[0, :, :].astype(np.uint16)  # Green channel (16-bit)
red_channel = frame_0[1, :, :].astype(np.uint16)  # Red channel (16-bit)

#%% Registration using Phase Cross-Correlation
# Calculate the shift between the two images
shift_value, error, diffphase = phase_cross_correlation(green_channel, red_channel, upsample_factor=10)

# Apply the translation to the red channel
registered_red_channel = shift(red_channel, shift=shift_value)

#%% Overlay after registration
# Create an RGB image for the overlay (black background)
overlay = np.zeros((green_channel.shape[0], green_channel.shape[1], 3))  # Start with black background
overlay[:, :, 1] = green_channel / np.max(green_channel)  # Add green channel to G
overlay[:, :, 0] = registered_red_channel / np.max(registered_red_channel)  # Add registered red channel to R

# Display the overlay after registration
plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.title('Overlay of Green and Registered Red Channels (Black Background)')
plt.axis('off')
plt.show()

#%% Save the registered first frame as multi-channel TIFF for Fiji
# Save only the first frame with registration applied
multi_channel_image_registered = np.stack((registered_red_channel, green_channel), axis=0)  # Stack channels (C, Y, X)

# Save the registered file as a stack with 'CYX' axis to ensure Fiji reads it as two channels
output_file_registered = '/Users/monikamakurath/Documents/pythonTestFiles/registered_first_frame_16bit.tiff'
tiff.imwrite(output_file_registered, multi_channel_image_registered.astype(np.uint16), photometric='minisblack', planarconfig='separate', metadata={'axes': 'CYX'})

print(f"Registered first frame (16-bit) saved at {output_file_registered}")
