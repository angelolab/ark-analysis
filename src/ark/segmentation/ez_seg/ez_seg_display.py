import pathlib
from typing import Union
from matplotlib.axes import Axes
from skimage.io import imread
from skimage.util import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from matploblib.figure import Figure
from matplotlib import gridspec
from alpineer import io_utils


# displaying a channel or composite
def display_channel_image(base_image_path: Union[str, pathlib.Path]) -> None:
    """
    Displays the base image.

    Args:
        base_image_path (Union[str, pathlib.Path]): The path to the image.
    """
    if isinstance(base_image_path, str):
        base_image_path = pathlib.Path(base_image_path)
    io_utils.validate_paths(base_image_path)

    # Load the base image
    base_image: np.ndarray = imread(base_image_path, as_gray=True)

    # Auto-scale the base image
    base_image_scaled = img_as_ubyte(base_image)

    # Display the original base image
    fig: Figure = plt.figure(dpi=300, figsize=(6, 6))
    fig.set_layout_engine(layout="constrained")
    gs = gridspec.GridSpec(1, 1, figure=fig)
    fig.suptitle(f"{base_image_path.name}")

    ax: Axes = fig.add_subplot(gs[0, 0])
    ax.imshow(base_image_scaled)

    ax.axis("off")

    plt.show()


# for displaying segmentation masks overlaid upon a base channel or composite
def overlay_mask_outlines(base_image_path, mask_image_path) -> None:
    """_summary_

    Args:
        base_image_path (_type_): _description_
        mask_image_path (_type_): _description_
    """
    if isinstance(base_image_path, str):
        base_image_path = pathlib.Path(base_image_path)
    if isinstance(mask_image_path, str):
        mask_image_path = pathlib.Path(mask_image_path)
    io_utils.validate_paths([base_image_path, mask_image_path])

    # Load the base image and mask image
    # Autoscale the base image
    base_image: np.ndarray = img_as_ubyte(image=imread(base_image_path, as_gray=True))
    mask_image: np.ndarray = imread(mask_image_path, as_gray=True)

    # Auto-scale the base image
    base_image_scaled = img_as_ubyte(base_image)

    # Apply Canny edge detection to extract outlines
    edges = cv2.Canny(mask_gray, 30, 100)

    # Create a copy of the base image to overlay the outlines
    overlay_image = np.copy(base_image)

    # Set the outline color to red
    outline_color = (0, 0, 255)

    # Overlay the outlines on the copy of the base image
    base_image_scaled[edges != 0] = outline_color

    # Display the original base image and the overlay image
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(base_image_scaled, cv2.COLOR_BGR2RGB))
    plt.title("Base Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# show all merged masks
def multiple_mask_displays(merge_mask_list, base_mask):
    # Create a grid to display the images
    num_images = len(merge_mask_list) * 2
    grid_cols = int(np.ceil(np.sqrt(num_images)))
    grid_rows = int(np.ceil(num_images / grid_cols))

    # Create the figure and subplots
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(10, 10))

    # Flatten the axs array to handle both cases (1-row grid and multi-row grid)
    axs = axs.ravel()

    # Iterate through the images and display them
    for i in range(num_images):
        # Read the image using cv2.imread()
        image = create_overlap_and_merge_visual(
            object_name=merge_mask_list[i], base_name=base_mask
        )

        # Display the image
        axs[i].imshow(image)
        axs[i].axis("off")

    # Adjust the spacing and layout
    fig.tight_layout()

    # Display the figure
    plt.show()


# for showing the overlap between two masks
def create_overlap_and_merge_visual(object_name, base_name):
    # read in masks
    object_name = cv2.imread(object_name, cv2.IMREAD_GRAYSCALE)
    base_name = cv2.imread(base_name, cv2.IMREAD_GRAYSCALE)
    merged_mask = cv2.imread(object_name + "_merged.tiff", cv2.IMREAD_GRAYSCALE)

    # Create an image with the size of the masks
    image = np.zeros((object_name.shape[0], object_name.shape[1], 3), dtype=np.uint8)

    # Assign colors to the non-overlapping areas of each mask
    image[object_name > 0] = (255, 0, 0)  # Blue for mask1
    image[base_name > 0] = (0, 0, 255)  # Red for mask2

    # Identify overlapping pixels and assign the white color
    overlap = np.logical_and(object_name == 255, base_name == 255)
    image[overlap] = (0, 0, 0)

    # Convert the merged mask image to grayscale
    merged_mask_gray = cv2.cvtColor(merged_mask, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to extract outlines
    edges = cv2.Canny(merged_mask_gray, 30, 100)

    # Set the outline color to green
    outline_color = (0, 255, 0)

    # Overlay the outlines on the copy of the base image
    image[edges != 0] = outline_color

    # return this image to the multi_merge_mask_display function.
    return image

    # # Display the resulting image
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
