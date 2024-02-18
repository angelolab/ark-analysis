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


def display_channel_image(
    base_image_path: Union[str, pathlib.Path],
    sub_folder_name: str,
    test_fov_name: str,
    channel_name: str,
    composite: bool = False,
) -> None:
    """
    Displays a channel or a composite image.

    Args:
        base_image_path (Union[str, pathlib.Path]): The path to the image.
        sub_folder_name (str): If a subfolder name for the channel data exists.
        test_fov_name (str): The name of the fov you wish to display.
        channel_name (str): The name of the channel you wish to display.
        composite (bool): Whether the image to be viewed is a composite image.
    """
    # Show test composite image
    if composite or (sub_folder_name is None):
        sub_folder_name = ""

    image_path = (
        pathlib.Path(base_image_path)
        / test_fov_name
        / sub_folder_name
        / f"{channel_name}.tiff"
    )

    if isinstance(image_path, str):
        image_path = pathlib.Path(image_path)
    io_utils.validate_paths(image_path)

    base_image: np.ndarray = imread(image_path, as_gray=True)

    # convert image to presentable RGB
    base_image_scaled = base_image / 255

    # Plot
    fig: Figure = plt.figure(dpi=300, figsize=(6, 6))
    fig.set_layout_engine(layout="constrained")
    gs = gridspec.GridSpec(1, 1, figure=fig)
    fig.suptitle(f"{image_path.name}")

    ax: Axes = fig.add_subplot(gs[0, 0])
    ax.imshow(base_image_scaled)
    ax.axis("off")


# for displaying segmentation masks overlaid upon a base channel or composite
def overlay_mask_outlines(
    fov: str,
    channel: str,
    image_dir: Union[str, os.PathLike],
    sub_folder_name: str,
    mask_name: str,
    mask_dir: Union[str, os.PathLike],
) -> None:
    """
    Displays a segmentation mask overlaid on a base image (channel or composite).

    Args:
        fov (str): name of fov to be viewed
        channel (str): name of channel to view
        image_dir (Union[str, os.PathLike]): The Path to channel for viewing.
        sub_folder_name (str): If a subfolder name for the channel data exists.
        mask_name (str): The name of mask to view
        mask_dir (Union[str, os.PathLike]): The path to the directory containing the mask.
    """
    if sub_folder_name is None:
        sub_folder_name = ""

    if isinstance(image_dir, str):
        image_dir = pathlib.Path(image_dir)
    if isinstance(mask_dir, str):
        mask_dir = pathlib.Path(mask_dir)

    image_dir = image_dir / sub_folder_name

    io_utils.validate_paths([image_dir, mask_dir])

    # Get ezseg and channel image paths
    channel_image_path = pathlib.Path(image_dir) / fov / f"{channel}.tiff"
    mask_image_path = pathlib.Path(mask_dir) / f"{fov}_{mask_name}.tiff"

    # Validate paths
    io_utils.validate_paths(paths=[channel_image_path, mask_image_path])

    # Load the base image and mask image
    # Autoscale the base image
    channel_image: np.ndarray = imread(channel_image_path, as_gray=True)
    mask_image: np.ndarray = imread(mask_image_path, as_gray=True)

    # convert image to presentable RGB
    channel_image_scaled = channel_image / 255

    # Apply Canny edge detection to extract outlines
    edges: np.ndarray = feature.canny(
        image=mask_image, low_threshold=0, high_threshold=1
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


def multiple_mask_display(
    fov: str,
    mask_name: str,
    object_mask_dir: Union[str, os.PathLike],
    cell_mask_dir: Union[str, os.PathLike],
    cell_mask_suffix: str,
    merged_mask_dir: Union[str, os.PathLike],
) -> None:
    """
    Create a grid to display the object, cell, and merged masks for a given fov.

    Args:
        fov (str): Name of the fov to view
        mask_name (str): Name of mask to view
        object_mask_dir (Union[str, os.PathLike]): Directory where the object masks are stored.
        cell_mask_dir (Union[str, os.PathLike]): Directory where the cell masks are stored.
        cell_mask_suffix (str): Suffix name of the cell mask files.
        merged_mask_dir (Union[str, os.PathLike]): Directory where the merged masks are stored.
    """
    if isinstance(object_mask_dir, str):
        object_mask_dir = pathlib.Path(object_mask_dir)
    if isinstance(cell_mask_dir, str):
        cell_mask_dir = pathlib.Path(cell_mask_dir)
    if isinstance(merged_mask_dir, str):
        merged_mask_dir = pathlib.Path(merged_mask_dir)
    io_utils.validate_paths([object_mask_dir, cell_mask_dir, merged_mask_dir])

    modified_overlay_mask: np.ndarray = create_overlap_and_merge_visual(
        fov, mask_name, object_mask_dir, cell_mask_dir, cell_mask_suffix, merged_mask_dir
    )

    # Create a new figure
    fig: Figure = plt.figure(dpi=300, figsize=(6, 6))
    fig.set_layout_engine(layout="constrained")
    gs = gridspec.GridSpec(1, 1, figure=fig)
    fig.suptitle(f"Merged Mask: {mask_name}")
    ax: Axes = fig.add_subplot(gs[0, 0])
    # Display color mask with transparency
    ax.imshow(modified_overlay_mask)
    ax.axis("off")


def create_overlap_and_merge_visual(
    fov: str,
    mask_name: str,
    object_mask_dir: pathlib.Path,
    cell_mask_dir: pathlib.Path,
    cell_mask_suffix: str,
    merged_mask_dir: pathlib.Path,
) -> np.ndarray:
    """
    Generate the NumPy Array representing the overlap between two masks

    Args:
        fov (str): Name of the fov to view
        mask_name (str): Name of mask to view
        object_mask_dir (pathlib.Path): Directory where the object masks are stored.
        cell_mask_dir (pathlib.Path): Directory where the cell masks are stored.
        cell_mask_suffix (str): Suffix name of the cell mask files.
        merged_mask_dir (pathlib.Path): Directory where the merged masks are stored.

    Returns:
        np.ndarray:
            Contains an overlap image of the two masks
    """
    # read in masks
    object_mask: np.ndarray = imread(object_mask_dir / f"{fov}_{mask_name}.tiff")
    cell_mask: np.ndarray = imread(
        cell_mask_dir / f"{fov}_{cell_mask_suffix}.tiff", as_gray=True
    )
    merged_mask: np.ndarray = imread(
        merged_mask_dir / f"{fov}_{mask_name}_merged.tiff", as_gray=True
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
