import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt

from skimage.io import imsave
from skimage.filters import sobel, threshold_multiotsu, meijering
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops_table
from skimage.exposure import equalize_adapthist

from ark.utils import io_utils, load_utils
from ark.utils.plot_utils import set_minimum_color_for_colormap
from ark import settings


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


def batch_segment_fibers(data_dir, fiber_channel, out_dir, img_sub_folder=None, batch_size=5,
                         **kwargs):
    """Segments fiber objects in batches

    Args:
        data_dir (str | PathLike):
            Folder containing dataset
        fiber_channel (str):
            Channel for fiber segmentation, e.g collagen.
        out_dir (str | PathLike):
            Directory to save fiber object labels and table.
        img_sub_folder (str | NoneType):
            Image subfolder name in `data_dir`. If there is not subfolder, set this to None.
        batch_size (int):
            Number of FoVs to load/process at a time
        **kwargs:
            Keyword arguments for `segment_fibers`
    """

    fovs = io_utils.list_folders(data_dir)
    batching_strategry = \
        [fovs[i:i + batch_size] for i in range(0, len(fovs), batch_size)]

    fiber_label_images = {}
    fiber_object_table = []

    for batch in batching_strategry:
        subset_xr = load_utils.load_imgs_from_tree(
            data_dir, img_sub_folder, fovs=batch, channels=[fiber_channel]
        )
        subtable, sublabel = segment_fibers(subset_xr, fiber_channel, out_dir, **kwargs)
        fiber_label_images.update(sublabel)
        fiber_object_table.append(subtable)

    fiber_object_table = pd.concat(fiber_object_table)

    return fiber_object_table, fiber_label_images


def segment_fibers(data_xr, fiber_channel, out_dir, blur=2, contrast_scaling_divisor=128,
                   fiber_widths=(2, 4), ridge_cutoff=0.1, sobel_blur=1, min_fiber_size=15,
                   object_properties=settings.FIBER_OBJECT_PROPS, debug=False):
    """ Segments fiber objects from image data

    Args:
        data_xr (xr.DataArray):
            Multiplexed image data in (fov, x, y, channel) format
        fiber_channel (str):
            Channel for fiber segmentation, e.g collagen.
        out_dir (str | PathLike):
            Directory to save fiber object labels and table.
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
        debug (bool):
            Save intermediate preprocessing steps

    Returns:
        pd.DataFrame, Dict[str, np.ndarray]:
         - Dataframe containing the fiber objects and their properties
         - Dictionary mapping fov names to their fiber label images
    """
    channel_xr = data_xr.loc[:, :, :, fiber_channel]
    fov_len = channel_xr.shape[1]

    fiber_label_images = {}
    fiber_object_table = []

    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"{out_dir} doesn't exist...")

    if debug:
        debug_path = os.path.join(out_dir, '_debug')
        os.makedirs(debug_path)

    for fov in channel_xr.fovs:
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
            imsave(os.path.join(debug_path, f'{fov}_thresholded.tiff'), threshed)
            imsave(os.path.join(debug_path, f'{fov}_ridges_thresholded.tiff'),
                   distance_transformed)
            imsave(os.path.join(debug_path, f'{fov}_meijering_filter.tiff'), ridges)
            imsave(os.path.join(debug_path, f'{fov}_contrast_adjusted.tiff'), contrast_adjusted)

        imsave(os.path.join(out_dir, f'{fov}_fiber_labels.tiff'), labeled_filtered)

        fiber_label_images[fov] = labeled_filtered

        fov_table = regionprops_table(labeled_filtered, properties=object_properties)

        fov_table = pd.DataFrame(fov_table)
        fov_table[settings.FOV_ID] = fov
        fiber_object_table.append(fov_table)

    fiber_object_table = pd.concat(fiber_object_table)
    fiber_object_table.to_csv(os.path.join(out_dir, 'fiber_object_table.csv'))

    return fiber_object_table, fiber_label_images
