import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import natsort as ns
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
from skimage.exposure import equalize_adapthist
from skimage.filters import meijering, sobel, threshold_multiotsu
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from alpineer import image_utils, io_utils, load_utils, misc_utils

from ark import settings
from ark.utils.plot_utils import set_minimum_color_for_colormap


def plot_fiber_segmentation_steps(data_dir, fov_name, fiber_channel, img_sub_folder=None, blur=2,
                                  contrast_scaling_divisor=128, fiber_widths=(2, 4),
                                  ridge_cutoff=0.1, sobel_blur=1, min_fiber_size=15,
                                  img_cmap=plt.cm.bone, labels_cmap=plt.cm.cool):
    """Plots output from each fiber segmentation step for single FoV

    Args:
        data_dir (str | PathLike):
            Folder containing dataset
        fov_name (str):
            Name of test FoV
        fiber_channel (str):
            Channel for fiber segmentation, e.g collagen
        img_sub_folder (str | NoneType):
            Whether to expect image subfolder in `data_dir`.  If no subfolder, set to None.
        blur (float):
            Preprocessing gaussian blur radius
        contrast_scaling_divisor (int):
            Roughly speaking, the average side length of a fibers bounding box.  This argument
            controls the local contrast enhancement operation, which helps differentiate dim
            fibers from only slightly more dim backgrounds.  This should always be a power of two.
        fiber_widths (Iterable):
            Widths of fibers to filter for.  Be aware that adding larger fiber widths can join
            close, narrow branches into one thicker fiber.
        ridge_cutoff (float):
            Threshold for ridge inclusion post-meijering filtering.
        sobel_blur (float):
            Gaussian blur radius for sobel driven elevation map creation
        min_fiber_size (int):
            Minimum area of fiber object
        img_cmap (matplotlib.cm.Colormap):
            Matplotlib colormap to use for (non-labeled) images
        labels_cmap (matplotlib.cm.Colormap):
            Base matplotlib colormap to use for labeled images.  This will only be applied to the
            non-zero labels, with the zero-region being colored black.
    """
    # no img_sub_folder, change to empty string to read directly from base folder
    if img_sub_folder is None:
        img_sub_folder = ""

    io_utils.validate_paths(data_dir)
    misc_utils.verify_in_list(fiber_channel=[fiber_channel],
                              all_channels=io_utils.remove_file_extensions(
                                  io_utils.list_files(
                                      os.path.join(data_dir, fov_name, img_sub_folder)
                                      )
                                  )
                              )

    data_xr = load_utils.load_imgs_from_tree(
        data_dir, img_sub_folder, fovs=[fov_name], channels=[fiber_channel]
    )

    channel_data = data_xr.loc[fov_name, :, :, fiber_channel].values

    _, axes = plt.subplots(3, 3)

    axes[0, 0].imshow(channel_data, cmap=img_cmap)
    axes[0, 0].set_title(f"{fov_name} {fiber_channel} raw image")

    blurred = ndi.gaussian_filter(channel_data.astype('float'), sigma=blur)
    axes[0, 1].imshow(blurred, cmap=img_cmap)
    axes[0, 1].set_title(f"Gaussian Blur, sigma={blur}")

    contrast_adjusted = equalize_adapthist(
        blurred / np.max(blurred),
        kernel_size=channel_data.shape[0] / contrast_scaling_divisor
    )
    axes[0, 2].imshow(contrast_adjusted, cmap=img_cmap)
    axes[0, 2].set_title(f"Contrast Adjuisted, CSD={contrast_scaling_divisor}")

    ridges = meijering(contrast_adjusted, sigmas=fiber_widths, black_ridges=False)
    axes[1, 0].imshow(ridges, cmap=img_cmap)
    axes[1, 0].set_title(f"Meijering Filter, fiber_widths={fiber_widths}")

    distance_transformed = ndi.gaussian_filter(
        distance_transform_edt(ridges > ridge_cutoff),
        sigma=1
    )
    axes[1, 1].imshow(distance_transformed, cmap=img_cmap)
    axes[1, 1].set_title(f"Ridges Filtered, ridge_cutoff={ridge_cutoff}")

    # watershed setup
    threshed = np.zeros_like(distance_transformed)
    thresholds = threshold_multiotsu(distance_transformed, classes=3)

    threshed[distance_transformed < thresholds[0]] = 1
    threshed[distance_transformed > thresholds[1]] = 2
    axes[1, 2].imshow(threshed, cmap=img_cmap)
    axes[1, 2].set_title("Watershed thresholding")

    elevation_map = sobel(
        ndi.gaussian_filter(distance_transformed, sigma=sobel_blur)
    )
    axes[2, 0].imshow(elevation_map, cmap=img_cmap)
    axes[2, 0].set_title(f"Sobel elevation map, sobel_blur={sobel_blur}")

    # build label color map
    transparent_cmap = set_minimum_color_for_colormap(labels_cmap)

    segmentation = watershed(elevation_map, threshed) - 1

    labeled, _ = ndi.label(segmentation)
    axes[2, 1].imshow(labeled, cmap=transparent_cmap)
    axes[2, 1].set_title("Unfiltered segmentation")

    labeled_filtered = remove_small_objects(labeled, min_size=min_fiber_size) * segmentation
    axes[2, 2].imshow(labeled_filtered, cmap=transparent_cmap)
    axes[2, 2].set_title(f"Filtered segmentation, min_fiber_size={min_fiber_size}")

    for ax in axes.reshape(-1):
        ax.axis('off')


def run_fiber_segmentation(data_dir, fiber_channel, out_dir, img_sub_folder=None,
                           csv_compression: Optional[Dict[str, str]] = None, **kwargs):
    """Segments fibers one FOV at a time

    Args:
        data_dir (str | PathLike):
            Folder containing dataset
        fiber_channel (str):
            Channel for fiber segmentation, e.g collagen.
        out_dir (str | PathLike):
            Directory to save fiber object labels and table.
        img_sub_folder (str | NoneType):
            Image subfolder name in `data_dir`. If there is not subfolder, set this to None.
        csv_compression (Optional[Dict[str, str]]): Dictionary of compression arguments to pass
            when saving csvs. See :meth:`to_csv <pandas.DataFrame.to_csv>` for details.
        **kwargs:
            Keyword arguments for `segment_fibers`

    Returns:
        pd.DataFrame:
         - Dataframe containing the fiber objects and their properties
    """

    # no img_sub_folder, change to empty string to read directly from base folder
    if img_sub_folder is None:
        img_sub_folder = ""

    io_utils.validate_paths([data_dir, out_dir])

    fovs = ns.natsorted(io_utils.list_folders(data_dir))
    misc_utils.verify_in_list(fiber_channel=[fiber_channel],
                              all_channels=io_utils.remove_file_extensions(
                                  io_utils.list_files(
                                      os.path.join(data_dir, fovs[0], img_sub_folder)
                                      )
                                  )
                              )

    fiber_object_table = []

    for fov in fovs:
        print(f'Processing FOV: {fov}')
        subset_xr = load_utils.load_imgs_from_tree(
            data_dir, img_sub_folder, fovs=fov, channels=[fiber_channel]
        )
        subtable = segment_fibers(subset_xr, fiber_channel, out_dir, fov, save_csv=False,
                                  **kwargs)
        fiber_object_table.append(subtable)

    fiber_object_table = pd.concat(fiber_object_table)
    fiber_object_table.to_csv(os.path.join(out_dir, 'fiber_object_table.csv'), index=False,
                              compression=csv_compression)

    return fiber_object_table


def segment_fibers(data_xr, fiber_channel, out_dir, fov, blur=2, contrast_scaling_divisor=128,
                   fiber_widths=(2, 4), ridge_cutoff=0.1, sobel_blur=1, min_fiber_size=15,
                   object_properties=settings.FIBER_OBJECT_PROPS, save_csv=True, debug=False):
    """ Segments fiber objects from image data

    Args:
        data_xr (xr.DataArray):
            Multiplexed image data in (fov, x, y, channel) format
        fiber_channel (str):
            Channel for fiber segmentation, e.g collagen.
        out_dir (str | PathLike):
            Directory to save fiber object labels and table.
        fov (str):
            name of the fov being processed
        blur (float):
            Preprocessing gaussian blur radius
        contrast_scaling_divisor (int):
            Roughly speaking, the average side length of a fibers bounding box.  This argument
            controls the local contrast enhancement operation, which helps differentiate dim
            fibers from only slightly more dim backgrounds.  This should always be a power of two.
        fiber_widths (Iterable):
            Widths of fibers to filter for.  Be aware that adding larger fiber widths can join
            close, narrow branches into one thicker fiber.
        ridge_cutoff (float):
            Threshold for ridge inclusion post-meijering filtering.
        sobel_blur (float):
            Gaussian blur radius for sobel driven elevation map creation
        min_fiber_size (int):
            Minimum area of fiber object
        object_properties (Iterable[str]):
            Properties to compute, any keyword for region props may be used.  Defaults are:
             - major_axis_length
             - minor_axis_length
             - orientation
             - centroid
             - label
             - eccentricity
             - euler_number
        save_csv (bool):
            Whether or not to save csv of fiber objects
        debug (bool):
            Save intermediate preprocessing steps

    Returns:
        pd.DataFrame:
         - Dataframe containing the fiber objects and their properties
    """
    channel_xr = data_xr.loc[:, :, :, fiber_channel]
    fov_len = channel_xr.shape[1]

    if debug:
        debug_path = os.path.join(out_dir, '_debug')
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)

    fiber_channel_data = channel_xr.loc[fov, :, :].values.astype('float')

    blurred = ndi.gaussian_filter(fiber_channel_data, sigma=blur)

    # local contrast enhancement
    contrast_adjusted = equalize_adapthist(
        blurred / np.max(blurred),
        kernel_size=fov_len / contrast_scaling_divisor
    )

    # meijering filtering
    ridges = meijering(contrast_adjusted, sigmas=fiber_widths, black_ridges=False)

    # remove image intensity influence for watershed setup
    distance_transformed = ndi.gaussian_filter(
        distance_transform_edt(ridges > ridge_cutoff),
        sigma=1
    )

    # watershed setup
    threshed = np.zeros_like(distance_transformed)
    thresholds = threshold_multiotsu(distance_transformed, classes=3)

    threshed[distance_transformed < thresholds[0]] = 1
    threshed[distance_transformed > thresholds[1]] = 2

    elevation_map = sobel(
        ndi.gaussian_filter(distance_transformed, sigma=sobel_blur)
    )

    segmentation = watershed(elevation_map, threshed) - 1

    labeled, _ = ndi.label(segmentation)

    labeled_filtered = remove_small_objects(labeled, min_size=min_fiber_size) * segmentation

    if debug:
        image_utils.save_image(os.path.join(debug_path, f'{fov}_thresholded.tiff'),
                               threshed)
        image_utils.save_image(os.path.join(debug_path, f'{fov}_ridges_thresholded.tiff'),
                               distance_transformed)
        image_utils.save_image(os.path.join(debug_path, f'{fov}_meijering_filter.tiff'),
                               ridges)
        image_utils.save_image(os.path.join(debug_path, f'{fov}_contrast_adjusted.tiff'),
                               contrast_adjusted)

    image_utils.save_image(os.path.join(out_dir, f'{fov}_fiber_labels.tiff'), labeled_filtered)

    fiber_object_table = regionprops_table(labeled_filtered, properties=object_properties)

    fiber_object_table = pd.DataFrame(fiber_object_table)
    fiber_object_table[settings.FOV_ID] = fov

    if save_csv:
        fiber_object_table.to_csv(os.path.join(out_dir, 'fiber_object_table.csv'))

    return fiber_object_table
