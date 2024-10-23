# Monika A. Makurath
# Sum intensities of registered image stack for each frame and save the summed stack
# libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# %% Load multi-channel registered image stack from specified path
# PC
stack_path = '/Users/makurathm/Documents/pythonTestFiles/registered_stack_16bit.tiff'  # Specify the path to the registered image stack
# laptop
# stack_path = '/Users/monikamakurath/Documents/pythonTestFiles/registered_stack_16bit.tiff'  # Specify the path to the registered image stack

# Load the multi-channel stack (assuming two channels: green and red)
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
# PC
output_file_summed_stack = '/Users/makurathm/Documents/pythonTestFiles/summed_stack_16bit.tiff'
# laptop
# output_file_summed_stack = '/Users/monikamakurath/Documents/pythonTestFiles/summed_stack_16bit.tiff'

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
