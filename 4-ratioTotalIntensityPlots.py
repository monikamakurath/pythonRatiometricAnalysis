# Monika A. Makurath
# Perform ratiometric analysis without normalization, display results, and plot ratiometric intensity over time
# libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from matplotlib import cm  # For applying colormap to save RGB image
import pandas as pd
import seaborn as sns

# Set seaborn styling
sns.set(style='whitegrid', context='talk', font_scale=1.2)

#%% Load the registered channels and the mask
# PC
stack_path = '/Users/makurathm/Documents/pythonTestFiles/registered_stack_16bit.tiff'  # Registered image stack path
mask_path = '/Users/makurathm/Documents/pythonTestFiles/cleaned_masks_stack.tiff'  # Mask path
# laptop
#stack_path = '/Users/monikamakurath/Documents/pythonTestFiles/registered_first_frame_16bit.tiff'  # Registered image stack path
#mask_path = '/Users/monikamakurath/Documents/pythonTestFiles/cleaned_mask.tiff'  # Mask path

# Load the registered image stack and mask
image_stack = tiff.imread(stack_path)
mask_stack = tiff.imread(mask_path)

# Split the two channels into two separate stacks
channel_1_stack = image_stack[:, 0, :, :]  # Extract the first channel
channel_2_stack = image_stack[:, 1, :, :]  # Extract the second channel

# Apply the mask to both channels
masked_channel_1_stack = channel_1_stack * mask_stack # green channel
masked_channel_2_stack = channel_2_stack * mask_stack # red channel

# Calculate total pixel intensity for each frame in both stacks
green_channel_intensities = np.sum(masked_channel_1_stack, axis=(1, 2))  # Sum over height and width for each frame
red_channel_intensities = np.sum(masked_channel_2_stack, axis=(1, 2))  # Sum over height and width for each frame
ratio_tot_intensity = green_channel_intensities/red_channel_intensities
ratio_tot_intensity_baseline_correction = ratio_tot_intensity - ratio_tot_intensity[0] + 1 # want to start at 1

# Ask the user to input the frame rate
frame_rate = float(input("Please enter the frame rate (time between frames in seconds): "))
# Create the frames array based on the number of frames and the input frame rate
num_frames = masked_channel_1_stack.shape[0]
frames = np.arange(num_frames) * frame_rate /60 # Adjust the frames array based on the frame rate (input in seconds but convert to minutes)

#%% save the data
# Set figure size and aspect ratio for square plots
plt.figure(figsize=(10, 6))  # Adjust the size for square format
plt.plot(frames, green_channel_intensities, label='Green Channel (Total Intensity)', color='green', linewidth=2)
plt.plot(frames, red_channel_intensities, label='Red Channel (Total Intensity)', color='red', linewidth=2)

# Set the labels, title, and legend to a more scientific style
plt.xlabel('Time (min)', fontsize=14)
plt.ylabel('Total Pixel Intensity', fontsize=14)
plt.title('Total Pixel Intensity for Each Frame (Green and Red Channels)', fontsize=16)
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()  # Ensure everything fits neatly
plt.savefig('/Users/makurathm/Documents/pythonTestFiles/total_pixel_intensity_plot.svg', format='svg', dpi=300)

# Second figure for intensity ratio
plt.figure(figsize=(10, 6))  # Square format
plt.plot(frames, ratio_tot_intensity_baseline_correction, label='Green/Red (Total Intensity)', color='black', linewidth=2)
plt.xlabel('Time (min)', fontsize=14)
plt.ylabel('Intensity Ratio (Green/Red)', fontsize=14)
plt.title('Green Total Intensity/Red Total Intensity', fontsize=16)
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()
plt.savefig('/Users/makurathm/Documents/pythonTestFiles/ratio_intensity_plot.svg', format='svg', dpi=300)


# Save data to an Excel spreadsheet
data = {
    'Time (s)': frames * 60,  # convert back to seconds
    'Green Channel Intensity': green_channel_intensities,
    'Red Channel Intensity': red_channel_intensities,
    'Intensity Ratio (Green/Red)': ratio_tot_intensity_baseline_correction
}

df = pd.DataFrame(data)
output_excel_path = '/Users/makurathm/Documents/pythonTestFiles/total_intensity_data.xlsx'
df.to_excel(output_excel_path, index=False)

print(f"Plots saved and data written to {output_excel_path}")