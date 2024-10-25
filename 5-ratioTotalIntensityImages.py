# Monika A. Makurath
# Perform ratiometric analysis without normalization, display results, and plot ratiometric intensity over time
# libraries
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from matplotlib import cm  # For colormap conversion to RGB

# %% Load the registered channels and the mask
# PC
stack_path = '/Users/makurathm/Documents/pythonTestFiles/registered_stack_16bit.tiff'  # Registered image stack path
mask_path = '/Users/makurathm/Documents/pythonTestFiles/cleaned_masks_stack.tiff'  # Mask path

# Load the registered image stack and mask
image_stack = tiff.imread(stack_path)
mask_stack = tiff.imread(mask_path)

# Split the two channels into two separate stacks
channel_1_stack = image_stack[:, 0, :, :]  # Extract the first channel
channel_2_stack = image_stack[:, 1, :, :]  # Extract the second channel

# Apply the mask to both channels
masked_channel_1_stack = channel_1_stack * mask_stack # green channel
masked_channel_2_stack = channel_2_stack * mask_stack # red channel


# Number of frames
num_frames = masked_channel_1_stack.shape[0]


# %% Function to compute ratiometric images for the entire stack (without normalization)
# shift = 0 # to display the values within the range that is visually pleasing
def compute_ratiometric_stack(green_stack, red_stack):
    epsilon = 1e-10  # Small constant to avoid division by zero
    ratiometric_stack = np.zeros_like(green_stack, dtype=np.float32)  # Initialize ratiometric stack

    for i in range(green_stack.shape[0]):  # Loop over each frame
        masked_green = green_stack[i]
        masked_red = red_stack[i]

        # Perform ratiometric analysis (Green / Red) for the current frame
        ratiometric_frame1 = masked_green / (masked_red + epsilon)
        #ratiometric_frame2 = ratiometric_frame1 - np.sum(ratiometric_frame1) + shift
        # Ensure shift is only added to non-zero and non-NaN values in ratiometric_frame2
        #ratiometric_frame2 = np.where((ratiometric_frame1 != 0) & (~np.isnan(ratiometric_frame1)),
                                   #   ratiometric_frame1 + shift, ratiometric_frame1)


        # Store the ratiometric frame in the stack
        ratiometric_stack[i] = ratiometric_frame1

    return ratiometric_stack


# %% Compute the ratiometric stack
ratiometric_stack = compute_ratiometric_stack(masked_channel_1_stack, masked_channel_2_stack)


# %% Convert ratiometric stack to RGB
def ratiometric_to_rgb(ratio_stack):
    rgb_stack = np.zeros((ratio_stack.shape[0], ratio_stack.shape[1], ratio_stack.shape[2], 3), dtype=np.uint8)

    for i in range(ratio_stack.shape[0]):
        norm_frame = np.clip(ratio_stack[i], 0, 3) / 3  # Clip between 0 and X for colormap
        rgb_frame = cm.jet(norm_frame)[:, :, :3]  # Apply 'jet' colormap and discard alpha channel
        rgb_stack[i] = (rgb_frame * 255).astype(np.uint8)  # Convert to 8-bit RGB values

    return rgb_stack


# Convert ratiometric stack to RGB
rgb_stack = ratiometric_to_rgb(ratiometric_stack)


# %% Display the ratiometric image for the first and last frames
def display_ratiometric_image(ratio_stack, frame_idx, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(ratio_stack[frame_idx], cmap='jet', vmin=0, vmax=5)  # Adjust vmin/vmax for better contrast
    plt.colorbar(label='Ratiometric Value (Green / Red)')
    plt.title(title)
    plt.axis('off')
    plt.show()


# Display first frame
display_ratiometric_image(rgb_stack, 0, 'Ratiometric Image - First Frame')

# Display last frame
display_ratiometric_image(rgb_stack, -1, 'Ratiometric Image - Last Frame')

# %% Save the entire ratiometric stack as a TIFF file
# PC
output_file_ratiometric_stack = '/Users/makurathm/Documents/pythonTestFiles/ratiometric_stack.tiff'

# Save the ratiometric stack (scale back to 16-bit range)
tiff.imwrite(output_file_ratiometric_stack, (ratiometric_stack * 65535).astype(np.uint16))  # Save as 16-bit TIFF stack

print(f"Ratiometric stack saved at {output_file_ratiometric_stack}")

# %% Save the RGB stack as high-resolution images (one per frame)
output_file_rgb_stack = '/Users/makurathm/Documents/pythonTestFiles/ratiometric_stack_rgb.tiff'

# Save the RGB stack as a high-resolution TIFF stack
tiff.imwrite(output_file_rgb_stack, rgb_stack)

print(f"RGB ratiometric stack saved at {output_file_rgb_stack}")
