# Monika A. Makurath
# Sum intensities of registered image stack for each frame and save the summed stack
# libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import getpass

#%% set file paths and load file
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
   # file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01.czi'
  #  file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-29-inNoGluGlutor/20240624-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-29-inNoGluGlutor.czi'
    #file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-15-Airyscan Processing-15/20240624-iGlucoSnFr2-JF646-Line-15-Airyscan Processing-15.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-16/20240624-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-16.czi'
  #  file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-30-inNoGluGlutor/20240624-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-30-inNoGluGlutor.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/20X/20240624-iGlucoSnFr2-JF646-Line-27-Airyscan Processing-25-control/20240624-iGlucoSnFr2-JF646-Line-27-Airyscan Processing-25-control.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/20X/20240624-iGlucoSnFr2-JF646-Line-29-Airyscan Processing-27-addingNoGluGlutor/20240624-iGlucoSnFr2-JF646-Line-29-Airyscan Processing-27-addingNoGluGlutor.czi'
   # file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/40x/20240624-iGlucoSnFr2-JF646-Line-22-Airyscan Processing-22/20240624-iGlucoSnFr2-JF646-Line-22-Airyscan Processing-22.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/40x/20240624-iGlucoSnFr2-JF646-Line-23-Airyscan Processing-23/20240624-iGlucoSnFr2-JF646-Line-23-Airyscan Processing-23.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/40x/20240624-iGlucoSnFr2-JF646-Line-25-Airyscan Processing-24/20240624-iGlucoSnFr2-JF646-Line-25-Airyscan Processing-24.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/63X/20240624-iGlucoSnFr2-JF646-Line-17-Airyscan Processing-17/20240624-iGlucoSnFr2-JF646-Line-17-Airyscan Processing-17.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/63X/20240624-iGlucoSnFr2-JF646-Line-18-Airyscan Processing-18/20240624-iGlucoSnFr2-JF646-Line-18-Airyscan Processing-18.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240822-hypothalamic-primary-cultures-ERiGlucoSnFR2/1 no treatment in DMEM/dish1/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01-Channel Alignment-02/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01-Channel Alignment-02.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240820-iGlucoSnFr2-JF646-Line-07-Airyscan Processing-09/20240820-iGlucoSnFr2-JF646-Line-07-Airyscan Processing-09.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240820-iGlucoSnFr2-JF646-Line-04-Airyscan Processing-05/20240820-iGlucoSnFr2-JF646-Line-04-Airyscan Processing-05.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240820-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-03/20240820-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-03.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240624-iGlucoSnFr2-JF646-Line-14-Airyscan Processing-07/20240624-iGlucoSnFr2-JF646-Line-14-Airyscan Processing-07.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240624-25-hypothalamic-primary-cultures-ERiGlucoSnFR2/well1-GLUTs/20240624-iGlucoSnFr2-JF646-Line-11-Airyscan Processing-03-addingDrugsGLUTs-zoom-p1/20240624-iGlucoSnFr2-JF646-Line-11-Airyscan Processing-03-addingDrugsGLUTs-zoom-p1.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240624-25-hypothalamic-primary-cultures-ERiGlucoSnFR2/well1-GLUTs/20240624-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-05-addingDrugs-one-cell-0ms/20240624-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-05-addingDrugs-one-cell-0ms.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240509-ERiGlucoSnFR-JF646-brain-cells-dish2B-LineScan/63X/from G to noGdrugs/20240509-ERiGlucoSnFR2-JF646-Line-06-Airyscan Processing-43-time-series-100-1s/20240509-ERiGlucoSnFR2-JF646-Line-06-Airyscan Processing-43-time-series-100-1s.czi'
    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240618-HepG2-ER.iGlucoSnFR2-JF646/20240618-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-03/20240618-iGlucoSnFr2-JF646-Line-02-Airyscan Processing-03.czi'


elif username == "monikamakurath":  # Laptop
    file_path = '/Users/monikamakurath/Documents/pythonTestFiles/test.czi'
else:
    raise ValueError("Unknown computer. Please specify the file path.")

path_no_file_name = os.path.dirname(file_path)
stack_path = os.path.join(path_no_file_name, 'registered_stack_16bit.tiff')
output_file_summed_stack = os.path.join(path_no_file_name, 'summed_stack_16bit.tiff')

#%% Load the multi-channel stack (assuming two channels: green and red)
multi_channel_stack = tiff.imread(stack_path)

# Get the number of frames (T) from the shape of the stack
num_frames = multi_channel_stack.shape[0]  # Shape is (T, C, Y, X)

# Initialize a list to store the summed frames
summed_stack = []

# Loop through each frame and sum the intensities of the green and red channels
for frame in range(num_frames):
    # Extract the green and red channels for the current frame
    green_channel = multi_channel_stack[frame, 0, :, :].astype(np.uint16)  # Green channel
    red_channel = multi_channel_stack[frame, 1, :, :].astype(np.uint16)  # Red channel

    # Sum the intensities of the two channels
    summed_image = green_channel + red_channel

    # Append the summed image to the stack
    summed_stack.append(summed_image)

# Convert the list of summed frames to a NumPy array (T, Y, X)
summed_stack = np.array(summed_stack)

# %% Save the summed stack as a multi-frame TIFF file
# Save the summed stack as a multi-frame TIFF (T, Y, X)
tiff.imwrite(output_file_summed_stack, summed_stack.astype(np.uint16), photometric='minisblack')

print(f"Summed stack (16-bit) saved at {output_file_summed_stack}")

# %% Display the Green, Red, and Summed Channels for the first frame
plt.figure(figsize=(18, 6))

# Display the Green Channel (first frame)
plt.subplot(1, 3, 1)
plt.imshow(multi_channel_stack[0, 0, :, :], cmap='gray')
plt.title('Green Channel (Frame 1)')
plt.colorbar()

# Display the Red Channel (first frame)
plt.subplot(1, 3, 2)
plt.imshow(multi_channel_stack[0, 1, :, :], cmap='gray')
plt.title('Red Channel (Frame 1)')
plt.colorbar()

# Display the Summed Image (first frame)
plt.subplot(1, 3, 3)
plt.imshow(summed_stack[0], cmap='gray')
plt.title('Summed Image (Green + Red, Frame 1)')
plt.colorbar()

plt.show()
