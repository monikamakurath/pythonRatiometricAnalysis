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

#%% load .czi file
# PC
file_path = '/Users/makurathm/Documents/pythonTestFiles/test.czi'  # Update this path
# laptop
# file_path = '/Users/monikamakurath/Documents/pythonTestFiles/test.czi'  # Update this path
czi = czifile.CziFile(file_path)
image = czifile.imread(file_path)

#%% initialize path to save
# PC
output_file_preregistration = '/Users/makurathm/Documents/pythonTestFiles/preregistration_image.tiff'  # Update with your desired path
# laptop
#output_file_preregistration = '/Users/monikamakurath/Documents/pythonTestFiles/preregistration_image.tiff'  # Update with your desired path

#%% show first frames
# Extract the first frame (T=0) for two channels
frame_0 = image[0, :, 0, :, :, 0]  # T=0, Z=0, S=0

# Channel 0 = Green, Channel 1 = Red (ensure green is first, red is second)
green_channel = frame_0[0, :, :]  # Green channel
red_channel = frame_0[1, :, :]  # Red channel (previously Far-Red)

# Create custom colormaps for green and red with white background
green_cmap = LinearSegmentedColormap.from_list('green_cmap', ['white', 'green'], N=256)
red_cmap = LinearSegmentedColormap.from_list('red_cmap', ['white', 'red'], N=256)

# Display the two channels side by side
plt.figure(figsize=(12, 6))

# Green channel (Green on White Background)
plt.subplot(1, 2, 1)
plt.imshow(green_channel, cmap=green_cmap)
plt.title('Green Channel')
plt.axis('off')

# Red channel (Red on White Background)
plt.subplot(1, 2, 2)
plt.imshow(red_channel, cmap=red_cmap)
plt.title('Red Channel')
plt.axis('off')

plt.show()

#%% save overlaid image for record of pre-registration
# Create an RGB image for the overlay (black background)
overlay = np.zeros((green_channel.shape[0], green_channel.shape[1], 3))  # Start with black background
overlay[:, :, 1] = green_channel / np.max(green_channel)  # Add green channel to G
overlay[:, :, 0] = red_channel / np.max(red_channel)  # Add red channel to R

# Display the overlay
plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.title('Overlay of Green and Red Channels (Black Background)')
plt.axis('off')
plt.show()

#%% Save as a multi-channel TIFF for Fiji (saving the two original channels as separate layers)
# Reverse the order of channels (flip green and red) before saving
multi_channel_image = np.stack((red_channel, green_channel), axis=0)  # Stack channels (C, Y, X) but swap them

# Save the file as a stack with 'CYX' axis to ensure Fiji reads it as two channels
tiff.imwrite(output_file_preregistration, multi_channel_image.astype(np.uint16), photometric='minisblack', planarconfig='separate', metadata={'axes': 'CYX'})

print(f"Multi-channel TIFF saved at {output_file_preregistration}")
