import itertools
import pathlib
from typing import Union
from matplotlib.axes import Axes
from skimage.io import imread
from skimage import feature, color
from skimage.util import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import gridspec
from alpineer import io_utils

def display_channel_image(base_image_path: Union[str, pathlib.Path]) -> None:
    """
    Displays a channel or a composite image.

    Args:
        base_image_path (Union[str, pathlib.Path]): The path to the image.
    """
    if isinstance(base_image_path, str):
        base_image_path = pathlib.Path(base_image_path)
    io_utils.validate_paths(base_image_path)

    base_image: np.ndarray = imread(base_image_path, as_gray=True)

    base_image_scaled = img_as_ubyte(base_image)

    # Plot
    fig: Figure = plt.figure(dpi=300, figsize=(6, 6))
    fig.set_layout_engine(layout="constrained")
    gs = gridspec.GridSpec(1, 1, figure=fig)
    fig.suptitle(f"{base_image_path.name}")

    ax: Axes = fig.add_subplot(gs[0, 0])
    ax.imshow(base_image_scaled)
    ax.axis("off")


# for displaying segmentation masks overlaid upon a base channel or composite
def overlay_mask_outlines(base_image_path, mask_image_path) -> None:
    """
    Displays a segmentation mask overlaid on a base image (channel or composite)

    Args:
        base_image_path (_type_): _description_
        mask_image_path (_type_): _description_
    """
    io_utils.validate_paths(paths=[base_image_path, mask_image_path])
    if isinstance(base_image_path, str):
        base_image_path = pathlib.Path(base_image_path)
    if isinstance(mask_image_path, str):
        mask_image_path = pathlib.Path(mask_image_path)

    # Load the base image and mask image
    # Autoscale the base image
    base_image: np.ndarray = imread(base_image_path, as_gray=True)
    mask_image: np.ndarray = imread(mask_image_path, as_gray=True)

    # Auto-scale the base image
    base_image_scaled = img_as_ubyte(base_image)

    # Apply Canny edge detection to extract outlines
    edges: np.ndarray = feature.canny(
        image=mask_image, low_threshold=30, high_threshold=100
    )

    # Set the outline color to red
    outline_color = (255, 0, 0)
    
    # Convert the base image to RGB
    rgb_base_image_scaled = color.gray2rgb(base_image_scaled)

    # Overlay the outlines on the copy of the base image
    rgb_base_image_scaled[edges != 0] = outline_color

    # Create a new figure
    fig: Figure = plt.figure(dpi=300, figsize=(6, 6))
    fig.set_layout_engine(layout="constrained")
    gs = gridspec.GridSpec(1, 1, figure=fig)
    fig.suptitle(f"{base_image_path.name}")
    ax: Axes = fig.add_subplot(gs[0, 0])
    ax.imshow(base_image)
    # Display color mask with transparency
    ax.imshow(rgb_base_image_scaled, alpha=0.3)
    ax.axis("off")

    fig.show()


def multiple_mask_displays(merge_mask_list, base_mask) -> None:
    # Create a grid to display the images
    num_images = len(merge_mask_list) * 2
    grid_cols = int(np.ceil(np.sqrt(num_images)))
    grid_rows = int(np.ceil(num_images / grid_cols))

    # Create the figure and subplots
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(10, 10))

    # Flatten the axs array to handle both cases (1-row grid and multi-row grid)
    axs = axs.ravel()

    # Iterate through the images and display them
    fig: Figure = plt.figure(dpi=300, figsize=(8, 8))
    fig.set_layout_engine(layout="constrained")
    gs = gridspec.GridSpec(nrows=grid_rows, ncols=grid_cols, figure=fig)

    for mask, (grix_row, grid_col) in zip(
        merge_mask_list, itertools.product(grid_rows, grid_cols)
    ):
        modified_mask = create_overlap_and_merge_visual(
            object_name=mask, base_name=base_mask
        )

        ax: Axes = fig.add_subplot(gs[grix_row, grid_col])

        # Display the mask
        ax.imshow(modified_mask)
        ax.axis("off")

    # Display the figure
    fig.show()


# for showing the overlap between two masks
def create_overlap_and_merge_visual(object_name, base_image_path) -> np.ndarray:
    # read in masks
    if isinstance(base_name_path, str):
        base_image_path = pathlib.Path(base_image_path)
    if isinstance(mask_image_path, str):
        mask_image_path = pathlib.Path(mask_image_path)
    io_utils.validate_paths([base_image_path, mask_image_path])

    object_name: np.ndarray = imread(object_name, as_gray=True)
    base_name: np.ndarray = imread(base_name, as_gray=True)
    merged_mask: np.ndarray = imread(object_name + "_merged.tiff", as_gray=True)

    # Create an image with the size of the masks
    image: np.ndarray = np.zeros(shape=(*object_name.shape, 3), dtype=np.uint8)

    # Assign colors to the non-overlapping areas of each mask
    image[object_name > 0] = (255, 0, 0)  # Blue for mask1
    image[base_name > 0] = (0, 0, 255)  # Red for mask2

    # Identify overlapping pixels and assign the white color
    overlap = np.logical_and(object_name == 255, base_name == 255)
    image[overlap] = (0, 0, 0)

    # Apply Canny edge detection to extract outlines
    edges = feature.canny(merged_mask, low_threshold=30, high_threshold=100)

    # Set the outline color to green
    outline_color = (0, 255, 0)

    # Overlay the outlines on the copy of the base image
    image[edges != 0] = outline_color

    # return this image to the multi_merge_mask_display function.
    return image
