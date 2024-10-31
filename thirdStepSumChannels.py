# third_step_summation.py

import numpy as np
import tifffile as tiff
import os
import matplotlib.pyplot as plt

def thirdStepSumChannels(output_dir):
    # Define paths for the registered stack and the output summed stack
    stack_path = os.path.join(output_dir, 'registered_stack_16bit.tiff')
    output_file_summed_stack = os.path.join(output_dir, 'summed_stack_16bit.tiff')

    # Load the registered multi-channel stack
    multi_channel_stack = tiff.imread(stack_path)
    num_frames = multi_channel_stack.shape[0]  # Get the number of frames (T)

    # Initialize a list to store the summed frames
    summed_stack = []

    # Loop through each frame and sum the intensities of the green and red channels
    for frame in range(num_frames):
        green_channel = multi_channel_stack[frame, 0, :, :].astype(np.uint16)
        red_channel = multi_channel_stack[frame, 1, :, :].astype(np.uint16)
        summed_image = green_channel + red_channel
        summed_stack.append(summed_image)

    # Convert the list of summed frames to a NumPy array (T, Y, X)
    summed_stack = np.array(summed_stack)

    # Save the summed stack as a multi-frame TIFF file
    tiff.imwrite(output_file_summed_stack, summed_stack.astype(np.uint16), photometric='minisblack')

    print(f"Summed stack (16-bit) saved at {output_file_summed_stack}")

    # Display the Green, Red, and Summed Channels for the first frame
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
