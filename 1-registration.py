# Monika A. Makurath
# Registration applied to first frame, then shift applied to all frames
# libraries
import czifile
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import tifffile as tiff
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

# %% Load .czi file
file_path = '/Users/makurathm/Documents/pythonTestFiles/test.czi'  # Update this path
czi = czifile.CziFile(file_path)
image = czifile.imread(file_path)

# Get image dimensions: (T, C, Z, Y, X, S)
num_frames = image.shape[0]  # Number of time frames
num_channels = image.shape[1]  # Number of channels (2 in this case)

# %% Step 1: Perform registration on the first frame only
# Extract the first frame (T=0) for two channels
first_frame = image[0, :, 0, :, :, 0]  # T=0, Z=0, S=0
green_channel_first_frame = first_frame[0, :, :].astype(np.uint16)  # Green channel (16-bit)
red_channel_first_frame = first_frame[1, :, :].astype(np.uint16)  # Red channel (16-bit)

# Calculate the shift between the two images (registration based on the first frame)
shift_value, error, diffphase = phase_cross_correlation(green_channel_first_frame, red_channel_first_frame,
                                                        upsample_factor=10)

# Apply the shift to the red channel of the first frame
registered_red_first_frame = shift(red_channel_first_frame, shift=shift_value)

# %% Step 2: Apply the same shift to all frames
# Initialize an empty list to hold all registered frames
registered_stack = []

# Loop through each frame and apply the same shift to the red channel
for frame in range(num_frames):
    # Extract the green and red channels for the current frame
    current_frame = image[frame, :, 0, :, :, 0]
    green_channel = current_frame[0, :, :].astype(np.uint16)
    red_channel = current_frame[1, :, :].astype(np.uint16)

    # Apply the calculated shift to the red channel for this frame
    registered_red_channel = shift(red_channel, shift=shift_value)

    # Stack the green and registered red channel, with green as the first channel
    registered_frame = np.stack((green_channel, registered_red_channel), axis=0)  # (C, Y, X) with green first

    # Add the registered frame to the stack
    registered_stack.append(registered_frame)

# Convert the list of registered frames to a 3D NumPy array (T, C, Y, X)
registered_stack = np.array(registered_stack)

# %% Step 3: Save the registered stack as a multi-frame TIFF for Fiji
# Save the registered stack as a multi-frame TIFF (T, C, Y, X) with green as first channel
output_file_registered_stack = '/Users/makurathm/Documents/pythonTestFiles/registered_stack_16bit.tiff'
tiff.imwrite(output_file_registered_stack, registered_stack.astype(np.uint16), photometric='minisblack',
             planarconfig='separate', metadata={'axes': 'TCYX'})

print(f"Registered image stack (16-bit) saved at {output_file_registered_stack}")
