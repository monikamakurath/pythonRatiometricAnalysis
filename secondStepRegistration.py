# second_step_registration.py

import czifile
import numpy as np
import tifffile as tiff
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import os

def secondStepRegistration(file_path, output_dir):
    # Load the .czi file (assuming it is already selected and passed as `file_path`)
    image = czifile.imread(file_path)

    # Get image dimensions
    num_frames = image.shape[0]  # Number of time frames

    # Initialize path for saving the registered stack
    output_file_registered_stack = os.path.join(output_dir, 'registered_stack_16bit.tiff')

    # Step 1: Perform registration on the first frame only
    first_frame = image[0, :, 0, :, :, 0]
    green_channel_first_frame = first_frame[0, :, :].astype(np.uint16)
    red_channel_first_frame = first_frame[1, :, :].astype(np.uint16)

    # Calculate shift for registration based on the first frame
    shift_value, error, diffphase = phase_cross_correlation(green_channel_first_frame, red_channel_first_frame, upsample_factor=10)

    # Step 2: Apply shift to all frames
    registered_stack = []
    for frame in range(num_frames):
        current_frame = image[frame, :, 0, :, :, 0]
        green_channel = current_frame[0, :, :].astype(np.uint16)
        red_channel = current_frame[1, :, :].astype(np.uint16)

        # Apply calculated shift to the red channel
        registered_red_channel = shift(red_channel, shift=shift_value)
        registered_frame = np.stack((green_channel, registered_red_channel), axis=0)
        registered_stack.append(registered_frame)

    # Convert list to NumPy array
    registered_stack = np.array(registered_stack)

    # Step 3: Save the registered stack
    tiff.imwrite(output_file_registered_stack, registered_stack.astype(np.uint16), photometric='minisblack',
                 planarconfig='separate', metadata={'axes': 'TCYX'})

    print(f"Registered image stack saved at {output_file_registered_stack}")
