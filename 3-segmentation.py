# Monika A. Makurath
# Apply mean threshold, remove small speckles, and save mask for each frame
# libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.morphology import remove_small_objects
import os
import getpass

# %% Load the previously saved summed stack
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
  #  file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-30-Airyscan Processing-28-time-series-after-NoGluAndGlutor/20240624-iGlucoSnFr2-JF646-Line-30-Airyscan Processing-28-time-series-after-NoGluAndGlutor.czi'
   # file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01.czi'
  #  file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-29-inNoGluGlutor/20240624-iGlucoSnFr2-JF646-Line-01-Airyscan Processing-29-inNoGluGlutor.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-15-Airyscan Processing-15/20240624-iGlucoSnFr2-JF646-Line-15-Airyscan Processing-15.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/10X/20240624-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-16/20240624-iGlucoSnFr2-JF646-Line-16-Airyscan Processing-16.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/20X/20240624-iGlucoSnFr2-JF646-Line-27-Airyscan Processing-25-control/20240624-iGlucoSnFr2-JF646-Line-27-Airyscan Processing-25-control.czi'
  #  file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/20X/20240624-iGlucoSnFr2-JF646-Line-29-Airyscan Processing-27-addingNoGluGlutor/20240624-iGlucoSnFr2-JF646-Line-29-Airyscan Processing-27-addingNoGluGlutor.czi'
   # file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/40x/20240624-iGlucoSnFr2-JF646-Line-22-Airyscan Processing-22/20240624-iGlucoSnFr2-JF646-Line-22-Airyscan Processing-22.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/40x/20240624-iGlucoSnFr2-JF646-Line-23-Airyscan Processing-23/20240624-iGlucoSnFr2-JF646-Line-23-Airyscan Processing-23.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/40x/20240624-iGlucoSnFr2-JF646-Line-25-Airyscan Processing-24/20240624-iGlucoSnFr2-JF646-Line-25-Airyscan Processing-24.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/63X/20240624-iGlucoSnFr2-JF646-Line-17-Airyscan Processing-17/20240624-iGlucoSnFr2-JF646-Line-17-Airyscan Processing-17.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240904-hypothalamic-slice-ERiGlucoSnFR2/63X/20240624-iGlucoSnFr2-JF646-Line-18-Airyscan Processing-18/20240624-iGlucoSnFr2-JF646-Line-18-Airyscan Processing-18.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240822-hypothalamic-primary-cultures-ERiGlucoSnFR2/1 no treatment in DMEM/dish1/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01-Channel Alignment-02/20240624-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-01-Channel Alignment-02.czi'
#    file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240820-iGlucoSnFr2-JF646-Line-07-Airyscan Processing-09/20240820-iGlucoSnFr2-JF646-Line-07-Airyscan Processing-09.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240820-iGlucoSnFr2-JF646-Line-04-Airyscan Processing-05/20240820-iGlucoSnFr2-JF646-Line-04-Airyscan Processing-05.czi'
 #   file_path = '/Volumes/jlslab/Monika/data/ER-glucose/ER-iGlucoSnFR2/20240820-hypothalamic-primary-cultures-ERiGlucoSnFR2/tunicamycin/20240820-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-03/20240820-iGlucoSnFr2-JF646-Line-03-Airyscan Processing-03.czi'
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
summed_stack_path = os.path.join(path_no_file_name, 'summed_stack_16bit.tiff')
output_file_masks_stack = os.path.join(path_no_file_name, 'cleaned_masks_stack.tiff')




# %% Load the multi-frame summed stack
summed_stack = tiff.imread(summed_stack_path)

# Get the number of frames (T) from the shape of the stack
num_frames = summed_stack.shape[0]  # Shape is (T, Y, X)

# Initialize a list to store the cleaned masks for each frame
cleaned_masks_stack = []

# Loop through each frame to create a mask
for frame in range(num_frames):
    # Extract the summed image for the current frame
    summed_image = summed_stack[frame, :, :].astype(np.uint16)

    # Step 1: Apply Mean Threshold to create a binary mask
    mean_threshold_value = np.mean(summed_image)  # Calculate mean intensity
    binary_mask = summed_image > mean_threshold_value  # Create binary mask

    # Step 2: Remove small speckles and background particles
    min_size = 500  # Adjust based on the size of your cells
    cleaned_mask = remove_small_objects(binary_mask, min_size=min_size)

    # Add the cleaned mask to the list
    cleaned_masks_stack.append(cleaned_mask)

# Convert the list of cleaned masks to a NumPy array (T, Y, X)
cleaned_masks_stack = np.array(cleaned_masks_stack).astype(np.uint16)

# %% Save the cleaned masks as a multi-frame TIFF file
# Save the cleaned mask stack as a multi-frame TIFF (T, Y, X)
tiff.imwrite(output_file_masks_stack, cleaned_masks_stack, photometric='minisblack')

print(f"Cleaned masks stack (16-bit) saved at {output_file_masks_stack}")

# %% Display the cleaned mask for the first frame as an example
plt.figure(figsize=(6, 6))
plt.imshow(cleaned_masks_stack[0, :, :], cmap='gray')
plt.title('Cleaned Mask (Frame 1)')
plt.axis('off')
plt.show()
