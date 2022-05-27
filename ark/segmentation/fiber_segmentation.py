import os

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt

from skimage.io import imsave
from skimage.filters import sobel, threshold_multiotsu, meijering
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops_table
from skimage.exposure import equalize_adapthist

from ark import settings


def segment_fibers(data_xr, fiber_channel, out_dir, blur=2, contrast_scaling_divisor=128,
                   fiber_widths=(2, 4), ridge_cutoff=0.1, sobel_blur=1, min_fiber_size=15,
                   object_properties=None, debug=False):
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
        object_properties (TBD):
            TBD
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

        # TODO: object_properties argument
        fov_table = regionprops_table(labeled_filtered, properties=[
            'major_axis_length',
            'minor_axis_length',
            'orientation',
            'centroid',
            'label',
            'eccentricity',
            'euler_number'
        ])

        fov_table = pd.DataFrame(fov_table)
        fov_table[settings.FOV_ID] = fov
        fiber_object_table.append(fov_table)

    fiber_object_table = pd.concat(fiber_object_table)
    fiber_object_table.to_csv(os.path.join(out_dir, 'fiber_object_table.csv'))

    return fiber_object_table, fiber_label_images
