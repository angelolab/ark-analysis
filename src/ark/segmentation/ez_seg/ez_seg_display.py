import pathlib
from typing import Union
from matplotlib.axes import Axes
from skimage.io import imread
from skimage import feature, color, filters
from skimage.util import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.figure import Figure
from matplotlib import gridspec
from alpineer import io_utils


def display_channel_image(base_image_path: Union[str, pathlib.Path], test_fov_name: str, channel_name: str) -> None:
    """
    Displays a channel or a composite image.

    Args:
        base_image_path (Union[str, pathlib.Path]): The path to the image.
        test_fov_name (str): The name of the fov you wish to display.
        channel_name (str): The name of the channel you wish to display.
    """
    # Show test composite image
    image_path = os.path.join(
        base_image_path, test_fov_name, channel_name + ".tiff"
    )

    if isinstance(image_path, str):
        image_path = pathlib.Path(image_path)
    io_utils.validate_paths(image_path)

    base_image: np.ndarray = imread(image_path, as_gray=True)

    base_image_scaled = img_as_ubyte(base_image)

    # Plot
    fig: Figure = plt.figure(dpi=300, figsize=(6, 6))
    fig.set_layout_engine(layout="constrained")
    gs = gridspec.GridSpec(1, 1, figure=fig)
    fig.suptitle(f"{image_path.name}")

    ax: Axes = fig.add_subplot(gs[0, 0])
    ax.imshow(base_image_scaled)
    ax.axis("off")


# for displaying segmentation masks overlaid upon a base channel or composite
def overlay_mask_outlines(fov_name, channel_to_view, channel_to_view_path, mask_name, mask_to_view_path) -> None:
    """
    Displays a segmentation mask overlaid on a base image (channel or composite)

    Args:
        fov_name (str): name of fov to be viewed
        channel_to_view (str): name of channel to view
        channel_to_view_path (str): path to channel to be viewed
        mask_name (str): name of mask to view
        mask_to_view_path (Union[str, pathlib.Path]): Path to mask to view
    """

    # Get test segmentation image paths
    channel_image_path = os.path.join(
        channel_to_view_path, fov_name, f"{channel_to_view}.tiff"
    )
    mask_image_path = os.path.join(
        mask_to_view_path, ''.join([f'{fov_name}', '_', mask_name, '.tiff'])
    )

    if isinstance(channel_to_view_path, str):
        channel_image_path = pathlib.Path(channel_image_path)
    if isinstance(mask_image_path, str):
        mask_image_path = pathlib.Path(mask_image_path)

    io_utils.validate_paths(paths=[channel_image_path, mask_image_path])

    # Load the base image and mask image
    # Autoscale the base image
    channel_image: np.ndarray = imread(channel_image_path, as_gray=True)
    mask_image: np.ndarray = imread(mask_image_path, as_gray=True)

    # Auto-scale the base image
    channel_image_scaled = img_as_ubyte(channel_image)

    # Apply Canny edge detection to extract outlines
    edges: np.ndarray = feature.canny(
        image=mask_image, low_threshold=30, high_threshold=100
    )

    # Set the outline color to red
    outline_color = (255, 0, 0)

    # Convert the base image to RGB
    rgb_channel_image_scaled = color.gray2rgb(channel_image_scaled)

    # Overlay the outlines on the copy of the base image
    rgb_channel_image_scaled[edges != 0] = outline_color

    # Create a new figure
    fig: Figure = plt.figure(dpi=300, figsize=(6, 6))
    fig.set_layout_engine(layout="constrained")
    gs = gridspec.GridSpec(1, 1, figure=fig)
    fig.suptitle(f"Mask: {mask_name}")
    ax: Axes = fig.add_subplot(gs[0, 0])
    ax.imshow(channel_image)
    # Display color mask with transparency
    ax.imshow(rgb_channel_image_scaled, alpha=0.3)
    ax.axis("off")


def multiple_mask_displays(test_fov_name, mask_name, object_mask_dir, cell_mask_dir, merged_mask_dir) -> None:
    """
    Create a grid to display the images

    Args:
        test_fov_name (str): name of fov to view
        mask_name (str): name of mask to view
        object_mask_dir (Posix.path): path name for object mask
        cell_mask_dir (Posix.path): path name for cell mask
        merged_mask_dir (Posix.path): path name for merged mask
    """

    modified_overlay_mask = create_overlap_and_merge_visual(test_fov_name, mask_name, object_mask_dir, cell_mask_dir,
                                                            merged_mask_dir)

    # Create a new figure
    fig: Figure = plt.figure(dpi=300, figsize=(6, 6))
    fig.set_layout_engine(layout="constrained")
    gs = gridspec.GridSpec(1, 1, figure=fig)
    fig.suptitle(f"Merged Mask: {mask_name}")
    ax: Axes = fig.add_subplot(gs[0, 0])
    # Display color mask with transparency
    ax.imshow(modified_overlay_mask)
    ax.axis("off")


def create_overlap_and_merge_visual(test_fov_name, mask_name, object_mask_dir, cell_mask_dir,
                                    merged_mask_dir) -> np.ndarray:
    """
    Show the overlap between two masks

    Args:
        test_fov_name (str): name of fov to view
        mask_name (str): name of mask to view
        object_mask_dir (Posix.path): path name for object mask
        cell_mask_dir (Posix.path): path name for cell mask
        merged_mask_dir (Posix.path): path name for merged mask

    Returns:
        np.ndarray:
            Contains an overlap image of the two masks
    """
    # read in masks
    io_utils.validate_paths([object_mask_dir, cell_mask_dir, merged_mask_dir])

    object_mask: np.ndarray = imread(
        object_mask_dir + "/" + test_fov_name + "_" + mask_name + ".tiff"
    )
    cell_mask: np.ndarray = imread(
        cell_mask_dir + "/" + test_fov_name + "_whole_cell.tiff", as_gray=True
    )
    merged_mask: np.ndarray = imread(
        merged_mask_dir + "/" + test_fov_name + "_" + mask_name + "_merged.tiff", as_gray=True
    )

    # Assign colors to the non-overlapping areas of each mask
    # Object masks in red
    red_array = np.zeros(shape=object_mask.shape, dtype=np.uint8)
    red_array[object_mask > 0] = 225

    # Cell masks in blue
    blue_array = np.zeros(shape=object_mask.shape, dtype=np.uint8)
    blue_array[cell_mask > 0] = 255

    # Merged mask edges in green
    merge_bool = merged_mask > 0
    edges = filters.sobel(merge_bool)
    green_array = np.zeros(shape=object_mask.shape, dtype=np.uint8)
    green_array[edges > 0] = 255

    # Combine red, green, and blue channels to create the final image
    image = np.stack([red_array, green_array, blue_array], axis=-1)

    # return this image to the multi_merge_mask_display function.
    return image
