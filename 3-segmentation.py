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
    file_path = '/Users/makurathm/Documents/pythonTestFiles/test.czi'
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
