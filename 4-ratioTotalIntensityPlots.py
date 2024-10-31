# Monika A. Makurath
# Perform ratiometric analysis without normalization, display results, and plot ratiometric intensity over time
# libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from matplotlib import cm  # For applying colormap to save RGB image
import pandas as pd
import seaborn as sns
import os
import getpass

# Set seaborn styling
sns.set(style='whitegrid', context='talk', font_scale=1.2)

#%% Load the registered channels and the mask
username = os.environ.get("USER") or getpass.getuser()

if username == "makurathm":  # Office computer
    #file_path = '/Users/makurathm/Documents/pythonTestFiles/test.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240913-HepG2-ERiGlucoSnFR2-DJN-Tunicamycin-Thepsigargin-BrefeldinA-Zeiss880/well5,7 Tunicamycin glucose/well7-Zeiss980-tunica-glucose-adding-noGluGlutor/20240914-well7-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-01-TNMglu-addingNoGluGlutor/20240914-well7-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-01-TNMglu-addingNoGluGlutor.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240913-HepG2-ERiGlucoSnFR2-DJN-Tunicamycin-Thepsigargin-BrefeldinA-Zeiss880/well6,8 Tunicamycin glucose-free/20240914-well8-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-02-TNMnoGlu-addingNoGluGlutor.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240909-HepG2-ERiGlucoSnFR2-DJN/10X - 36h later/2 glutor/20240618-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-25-addingRegularDMEM-to-RegularDMEM/20240618-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-25-addingRegularDMEM-to-RegularDMEM.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240909-HepG2-ERiGlucoSnFR2-DJN/10X - 36h later/2 glutor/20240618-iGlucoSnFr2-JF646-Line-04-Airyscan Processing-26-addingNoGluDMEM-to-no-treatment-regularDMEM/20240618-iGlucoSnFr2-JF646-Line-04-Airyscan Processing-26-addingNoGluDMEM-to-no-treatment-regularDMEM.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240909-HepG2-ERiGlucoSnFR2-DJN/10X - 36h later/2 glutor/20240618-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-24-addingRegularDMEMwithGlutor/20240618-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-24-addingRegularDMEMwithGlutor.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240909-HepG2-ERiGlucoSnFR2-DJN/10X - 36h later/2 glutor/20240911-iGlucoSnFr2-JF646-Line-09-Airyscan Processing-17-from-no-treatment-toNoGluGlutor/20240911-iGlucoSnFr2-JF646-Line-09-Airyscan Processing-17-from-no-treatment-toNoGluGlutor.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240909-HepG2-ERiGlucoSnFR2-DJN/10X - 36h later/2 glutor/20240911-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-22-addingNoGluGlutor-to-regularDMEM-repeat-diffPowerSettings/20240911-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-22-addingNoGluGlutor-to-regularDMEM-repeat-diffPowerSettings.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240909-HepG2-ERiGlucoSnFR2-DJN/10X - 36h later/2 glutor/20240911-iGlucoSnFr2-JF646-Line-08-Airyscan Processing-16-DNJnoGlu-addingNoGluGlutor/20240911-iGlucoSnFr2-JF646-Line-08-Airyscan Processing-16-DNJnoGlu-addingNoGluGlutor.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240909-HepG2-ERiGlucoSnFR2-DJN/10X - 36h later/1/20240618-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-07-NoGluDJN/20240618-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-07-NoGluDJN.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240913-HepG2-ERiGlucoSnFR2-DJN-Tunicamycin-Thepsigargin-BrefeldinA-Zeiss880/well6,8 Tunicamycin glucose-free/20240914-well8-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-02-TNMnoGlu-addingNoGluGlutor/20240914-well8-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-02-TNMnoGlu-addingNoGluGlutor.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240913-HepG2-ERiGlucoSnFR2-DJN-Tunicamycin-Thepsigargin-BrefeldinA-Zeiss880/well6,8 Tunicamycin glucose-free/20240618-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-02-TNMnoGlu-addingNoGluGlutor/20240618-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-02-TNMnoGlu-addingNoGluGlutor.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/2.5X/20240624-iGlucoSnFr2-JF646-Line-05-Airyscan Processing-03-Channel Alignment-05/20240624-iGlucoSnFr2-JF646-Line-05-Airyscan Processing-03-Channel Alignment-05.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/2.5X/20240624-iGlucoSnFr2-JF646-Line-11-Airyscan Processing-10/20240624-iGlucoSnFr2-JF646-Line-11-Airyscan Processing-10.czi'
   # file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-30-Airyscan Processing-28-time-series-after-NoGluAndGlutor/20240624-iGlucoSnFr2-JF646-Line-30-Airyscan Processing-28-time-series-after-NoGluAndGlutor.czi'
  #  file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01.czi'
  #  file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-29-inNoGluGlutor/20240624-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-29-inNoGluGlutor.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-15-Airyscan Processing-15/20240624-iGlucoSnFr2-JF646-Line-15-Airyscan Processing-15.czi'
   # file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-16/20240624-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-16.czi'
  #  file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-30-inNoGluGlutor/20240624-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-30-inNoGluGlutor.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/20X/20240624-iGlucoSnFr2-JF646-Line-27-Airyscan Processing-25-control/20240624-iGlucoSnFr2-JF646-Line-27-Airyscan Processing-25-control.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/20X/20240624-iGlucoSnFr2-JF646-Line-29-Airyscan Processing-27-addingNoGluGlutor/20240624-iGlucoSnFr2-JF646-Line-29-Airyscan Processing-27-addingNoGluGlutor.czi'
   # file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/40x/20240624-iGlucoSnFr2-JF646-Line-22-Airyscan Processing-22/20240624-iGlucoSnFr2-JF646-Line-22-Airyscan Processing-22.czi'
  #  file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/40x/20240624-iGlucoSnFr2-JF646-Line-23-Airyscan Processing-23/20240624-iGlucoSnFr2-JF646-Line-23-Airyscan Processing-23.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/40x/20240624-iGlucoSnFr2-JF646-Line-25-Airyscan Processing-24/20240624-iGlucoSnFr2-JF646-Line-25-Airyscan Processing-24.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/63X/20240624-iGlucoSnFr2-JF646-Line-17-Airyscan Processing-17/20240624-iGlucoSnFr2-JF646-Line-17-Airyscan Processing-17.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/63X/20240624-iGlucoSnFr2-JF646-Line-18-Airyscan Processing-18/20240624-iGlucoSnFr2-JF646-Line-18-Airyscan Processing-18.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240822-hypothalamic-primary-cultures-ERiGlucoSnFR2/1 no treatment in DMEM/dish1/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01-Channel Alignment-02/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01-Channel Alignment-02.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240820-iGlucoSnFr2-JF646-Line-07-Airyscan Processing-09/20240820-iGlucoSnFr2-JF646-Line-07-Airyscan Processing-09.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240820-iGlucoSnFr2-JF646-Line-04-Airyscan Processing-05/20240820-iGlucoSnFr2-JF646-Line-04-Airyscan Processing-05.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240820-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-03/20240820-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-03.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240820-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-03/20240820-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-03.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240624-iGlucoSnFr2-JF646-Line-14-Airyscan Processing-07/20240624-iGlucoSnFr2-JF646-Line-14-Airyscan Processing-07.czi'
   # file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240624-25-hypothalamic-primary-cultures-ERiGlucoSnFR2/well1-GLUTs/20240624-iGlucoSnFr2-JF646-Line-11-Airyscan Processing-03-addingDrugsGLUTs-zoom-p1/20240624-iGlucoSnFr2-JF646-Line-11-Airyscan Processing-03-addingDrugsGLUTs-zoom-p1.czi'
  #  file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240624-25-hypothalamic-primary-cultures-ERiGlucoSnFR2/well1-GLUTs/20240624-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-05-addingDrugs-one-cell-0ms/20240624-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-05-addingDrugs-one-cell-0ms.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240509-ERiGlucoSnFR-JF646-brain-cells-dish2B-LineScan/63X/from G to noGdrugs/20240509-ERiGlucoSnFR2-JF646-Line-06-Airyscan Processing-43-time-series-100-1s/20240509-ERiGlucoSnFR2-JF646-Line-06-Airyscan Processing-43-time-series-100-1s.czi'
    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240618-HepG2-ER.iGlucoSnFR2-JF646/20240618-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-03/20240618-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-03.czi'

elif username == "monikamakurath":  # Laptop
    file_path = '/Users/monikamakurath/Documents/pythonTestFiles/test.czi'
else:
    raise ValueError("Unknown computer. Please specify the file path.")

path_no_file_name = os.path.dirname(file_path)
stack_path = os.path.join(path_no_file_name, 'registered_stack_16bit.tiff')
mask_path = os.path.join(path_no_file_name, 'cleaned_masks_stack.tiff')

# output save paths
output_file_plot_raw_int_svg = os.path.join(path_no_file_name, 'total_pixel_intensity_plot.svg')
output_file_plot_ratio_int_svg = os.path.join(path_no_file_name, 'ratio_intensity_plot.svg')
output_file_plot_ratio_int_combined_svg = os.path.join(path_no_file_name, 'ratio_intensity_combined_plot.svg')
output_excel_path = os.path.join(path_no_file_name, 'total_intensity_data.xlsx')

# %% MANUALLY SET FRAME RATE!
frame_rate = 9.65    # in seconds
print('Remember to update the frame rate and pixel size.')
# Ask the user to input the frame rate
#frame_rate = float(input("Please enter the frame rate (time between frames in seconds): "))

# %% Load the registered image stack and mask
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

# %% Create the frames array based on the number of frames and the input frame rate
num_frames = masked_channel_1_stack.shape[0]
frames = np.arange(num_frames) * frame_rate /60 # Adjust the frames array based on the frame rate (input in seconds but convert to minutes)

#%% save the two plots seperately
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
plt.savefig(output_file_plot_raw_int_svg, format='svg', dpi=300)

# Second figure for intensity ratio
plt.figure(figsize=(10, 6))  # Square format
plt.plot(frames, ratio_tot_intensity_baseline_correction, label='Green/Red (Total Intensity)', color='black', linewidth=2)
plt.xlabel('Time (min)', fontsize=14)
plt.ylabel('Intensity Ratio (Green/Red)', fontsize=14)
plt.title('Green Total Intensity/Red Total Intensity', fontsize=16)
plt.legend(frameon=False, fontsize=12)
plt.ylim(0,2)
plt.tight_layout()
plt.savefig(output_file_plot_ratio_int_svg, format='svg', dpi=300)


# %% create a subplot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 16))  # Adjust height for square aspect ratio # Create a figure with two square subplots, stacked vertically

# First subplot for total intensity of green and red channels
ax1.plot(frames, green_channel_intensities, label='Green Channel (Total Intensity)', color='green', linewidth=2)
ax1.plot(frames, red_channel_intensities, label='Red Channel (Total Intensity)', color='red', linewidth=2)
ax1.set_xlabel('Time (min)', fontsize=14)
ax1.set_ylabel('Total Pixel Intensity', fontsize=14)
ax1.set_title('Total Pixel Intensity for Each Frame (Green and Red Channels)', fontsize=16)
ax1.legend(frameon=False, fontsize=12)

# Second subplot for intensity ratio
ax2.plot(frames, ratio_tot_intensity_baseline_correction, label='Green/Red (Total Intensity)', color='black', linewidth=2)
ax2.set_xlabel('Time (min)', fontsize=14)
ax2.set_ylabel('Intensity Ratio (Green/Red)', fontsize=14)
ax2.set_title('Green Total Intensity/Red Total Intensity', fontsize=16)
ax2.legend(frameon=False, fontsize=12)

# Adjust layout to ensure subplots are neat and save the figure
plt.tight_layout()
plt.savefig(output_file_plot_ratio_int_combined_svg, format='svg', dpi=300)

# %% format the ratio plot
# Second figure for intensity ratio
plt.figure(figsize=(10, 10))  # Square format
plt.plot(frames, ratio_tot_intensity_baseline_correction, label='', color='midnightblue', linewidth=4)
plt.xlabel('Time (min)', fontsize=24)
plt.ylabel('Intensity Ratio (Green/Red)', fontsize=24)
#plt.title('Green Total Intensity/Red Total Intensity', fontsize=16)
plt.legend(frameon=False, fontsize=18)
plt.ylim(0, 2)  # Set y-axis limits for the first subplot

# remove grid
plt.grid(True)

# Remove top and right spines (frame lines)
ax = plt.gca()  # Get current axes
ax.spines['top'].set_visible(True)     # change to False
ax.spines['right'].set_visible(True)     # change to False

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(output_file_plot_ratio_int_svg, format='svg', dpi=300)


# %% save to an excel spreadsheet
# Save data to an Excel spreadsheet
data = {
    'Time (s)': frames * 60,  # convert back to seconds
    'Raw Green Channel Intensity': green_channel_intensities,
    'Raw Red Channel Intensity': red_channel_intensities,
    'Raw Intensity Ratio (Green/Red)': ratio_tot_intensity
}

df = pd.DataFrame(data)
df.to_excel(output_excel_path, index=False)

print(f"Plots saved and data written to {output_excel_path}")


