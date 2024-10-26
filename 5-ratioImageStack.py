# Monika A. Makurath
# Perform ratiometric analysis without normalization, display results, and plot ratiometric intensity over time
# libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
import os
import getpass
import math
from PIL import Image
import matplotlib.cm as cm

plt.rcParams['font.family'] = 'Arial'   # set the font for all images 

# %% Load the registered channels and the mask
username = os.environ.get("USER") or getpass.getuser()

if username == "makurathm":  # Office computer
    file_path = '/Users/makurathm/Documents/pythonTestFiles/test.czi'
elif username == "monikamakurath":  # Laptop
    file_path = '/Users/monikamakurath/Documents/pythonTestFiles/test.czi'
else:
    raise ValueError("Unknown computer. Please specify the file path.")

path_no_file_name = os.path.dirname(file_path)
stack_path = os.path.join(path_no_file_name, 'registered_stack_16bit.tiff')
mask_path = os.path.join(path_no_file_name, 'cleaned_masks_stack.tiff')

# output save paths
output_file_rgb_stack = os.path.join(path_no_file_name, 'ratiometric_stack.tiff')

# %% Load the registered image stack and mask
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
def compute_ratiometric_stack(green_stack, red_stack):
    epsilon = 1e-10  # Small constant to avoid division by zero
    ratiometric_stack = np.zeros_like(green_stack, dtype=np.float32)  # Initialize ratiometric stack

    for i in range(green_stack.shape[0]):  # Loop over each frame
        masked_green = green_stack[i]   # background values are zeros
        masked_red = red_stack[i]   # background values are zeros

        # Perform ratiometric analysis (Green / Red) for the current frame
        ratiometric_frame1 = masked_green / (masked_red + epsilon)

        # Store the ratiometric frame in the stack
        ratiometric_stack[i] = ratiometric_frame1

    return ratiometric_stack


# %% Compute the ratiometric stack
ratiometric_stack = compute_ratiometric_stack(masked_channel_1_stack, masked_channel_2_stack)

# %% Manually input the frame rate and the scale bar
num_frames = len(ratiometric_stack)     # Total frames
# Frame rate in seconds
framerate_seconds = 59.86   # Frame rate in seconds
# Convert frame rate to minutes
framerate_minutes = framerate_seconds / 60.0

# Construct the time array
time_array = np.arange(num_frames) * framerate_minutes

# Generate text size based on figure size
text_size = plt.gcf().get_size_inches()[1] / 30

# %% Manually input pixel size for the scale bar and scale bar location parameters
pixel_width = 0.1322914     # pixel resolution in microns
pixel_height = 0.1322914    # pixel resolution in microns
scale_bar_length_microns = 100      # desired scale bar size in microns
scale_bar_length_pixels = scale_bar_length_microns / pixel_width     # Calculate scale bar size in pixels

# Scale bar location parameters
scale_bar_height_pixels = 25  # height of scale bar
scale_bar_vertical_padding = 100  # padding from bottom
scale_bar_horizontal_padding = 100  # padding from right

# %% Clip top and bottom pixel values
clipped_stack = np.copy(ratiometric_stack)      # Initialize a new stack to store the clipped frames

# %% Loop through all the frames in the stack and create pretty images
for i, frame in enumerate(ratiometric_stack):

    # Filter out background pixels
    non_bg_pixels = frame[frame > 0]

    # Continue only if there are non-background pixels in the frame
    if non_bg_pixels.size > 0:

        # Calculate clip values
        clip_min, clip_max = np.percentile(non_bg_pixels, [0.01, 99])  # clip at 1st and 99th percentiles

        # Apply the clipping
        non_bg_pixels_clipped = np.clip(non_bg_pixels, a_min=clip_min, a_max=clip_max)
        frame[frame > 0] = non_bg_pixels_clipped

        # Save the clipped frame back to the stack
        clipped_stack[i] = frame

        # Plot histograms for first and last frames
        if i == 0 or i == len(ratiometric_stack) - 1:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            # Plot original histogram
            axs[0].hist(non_bg_pixels, bins=256, color='gray')
            axs[0].set_title(f'Frame {i + 1}: Original')

            # Plot clipped histogram
            axs[1].hist(non_bg_pixels_clipped, bins=256, color='gray')
            axs[1].set_title(f'Frame {i + 1}: Clipped')

            plt.show()


# %% Display the first frame
plt.figure(figsize=[6, 6])
plt.imshow(clipped_stack[0], cmap='turbo', vmin=0, vmax=3)
plt.axis('off')  # to hide axis

# Add time text to the plot
plt.text(0.02, 0.95, '{} min'.format(math.ceil(time_array[0])), transform=plt.gca().transAxes, fontsize=14,
         verticalalignment='top', color='w')

# Calculate the position of scale bar using image dimensions and scale bar dimensions
scale_bar_x_position = clipped_stack[0].shape[1] - scale_bar_length_pixels - scale_bar_horizontal_padding
scale_bar_y_position = clipped_stack[0].shape[0] - scale_bar_height_pixels - scale_bar_vertical_padding

# Create scale bar patch
scale_bar = patches.Rectangle((scale_bar_x_position, scale_bar_y_position), scale_bar_length_pixels,
                              scale_bar_height_pixels, edgecolor='white', facecolor='white')

# Draw scale bar
plt.gca().add_patch(scale_bar)

# Display the plot
plt.show()
# %% Display and save the last frame with color bar
output_file_last_frame = os.path.join(path_no_file_name, 'last_frame_high_res.tiff')    # Define the path to save the high-resolution TIFF file
output_file_last_frame_svg = os.path.join(path_no_file_name, 'last_frame_high_res_svg.svg')    # Define the path to save the high-resolution TIFF file

# Set high DPI and larger figure size for higher resolution
high_dpi = 300  # Increase for higher resolution

# Create the figure and apply settings
plt.figure(figsize=[8, 8], dpi=high_dpi)
plt.imshow(clipped_stack[-1], cmap='turbo', vmin=0, vmax=3)
plt.title('Last Frame')
plt.axis('off')  # Hide axes
plt.colorbar()

# Add time text to the plot
plt.text(0.02, 0.95, '{} min'.format(math.ceil(time_array[-1])), transform=plt.gca().transAxes, fontsize=18,
         verticalalignment='top', color='white')

# Calculate the position of scale bar using image dimensions and scale bar dimensions
scale_bar_x_position = clipped_stack[0].shape[1] - scale_bar_length_pixels - scale_bar_horizontal_padding
scale_bar_y_position = clipped_stack[0].shape[0] - scale_bar_height_pixels - scale_bar_vertical_padding

# Create and add scale bar patch
scale_bar = patches.Rectangle((scale_bar_x_position, scale_bar_y_position), scale_bar_length_pixels,
                              scale_bar_height_pixels, edgecolor='white', facecolor='white')
plt.gca().add_patch(scale_bar)

# Save the figure as a high-resolution TIFF without additional padding
plt.savefig(output_file_last_frame, format='tiff', dpi=high_dpi, bbox_inches='tight', pad_inches=0)
plt.savefig(output_file_last_frame_svg, format='tiff', dpi=high_dpi, bbox_inches='tight', pad_inches=0)
plt.close()  # Close to free up memory



print(f"Last frame saved as high-resolution TIFF at {output_file_last_frame}")

# %% Loop through all frames and save them
high_dpi = 300  # Adjust this value as needed for quality. Set the desired DPI for high-resolution images

# List to store each frame with annotations
formatted_frames = []

# Loop through each frame and add annotations
for i, frame in enumerate(clipped_stack):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=high_dpi)  # Set figure DPI
    ax.imshow(frame, cmap='turbo', vmin=0, vmax=3)
    ax.axis('off')  # Hide axes for clean display

    # Add time text
    ax.text(0.02, 0.95, f'{math.ceil(time_array[i])} min', transform=ax.transAxes, fontsize=18, color='white',
            verticalalignment='top')

    # Add scale bar
    scale_bar_x_position = frame.shape[1] - scale_bar_length_pixels - scale_bar_horizontal_padding
    scale_bar_y_position = frame.shape[0] - scale_bar_height_pixels - scale_bar_vertical_padding
    scale_bar = patches.Rectangle((scale_bar_x_position, scale_bar_y_position), scale_bar_length_pixels,
                                  scale_bar_height_pixels, edgecolor='white', facecolor='white')
    ax.add_patch(scale_bar)

    # Adjust the subplot parameters to remove any padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save frame to an image array without the white frame
    fig.canvas.draw()
    rgba_buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba_frame = rgba_buffer.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    rgb_frame = rgba_frame[:, :, :3]  # Retain only RGB channels
    formatted_frames.append(rgb_frame)
    plt.close(fig)  # Close figure to save memory

# Save all formatted frames into a single TIFF stack
output_file_rgb_stack = os.path.join(path_no_file_name, 'ratiometric_stack.tiff')
tiff.imwrite(output_file_rgb_stack, np.array(formatted_frames), photometric='rgb')
print(f"TIFF stack saved successfully to {output_file_rgb_stack}.")

# Create and save the color bar separately in both orientations without padding, and at high DPI
# Horizontal color bar
fig, ax = plt.subplots(figsize=(6, 1), dpi=high_dpi)
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='turbo'), cax=ax, orientation='horizontal')
cbar.set_label('Intensity')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
color_bar_horizontal_path = os.path.join(path_no_file_name, 'color_bar_horizontal.svg')
plt.savefig(color_bar_horizontal_path, dpi=high_dpi, bbox_inches='tight', pad_inches=0)
plt.close(fig)

# Vertical color bar
fig, ax = plt.subplots(figsize=(1, 6), dpi=high_dpi)
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='turbo'), cax=ax, orientation='vertical')
cbar.set_label('Intensity')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
color_bar_vertical_path = os.path.join(path_no_file_name, 'color_bar_vertical.svg')
plt.savefig(color_bar_vertical_path, dpi=high_dpi, bbox_inches='tight', pad_inches=0)
plt.close(fig)

print(f"Color bars saved as {color_bar_horizontal_path} and {color_bar_vertical_path}.")