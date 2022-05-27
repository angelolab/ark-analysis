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


# TODO: debug outputs
def segment_fibers(data_xr, fiber_channel, out_dir, blur=2, contrast_scaling_divisor=128,
                   fiber_widths=(2, 4, 6), ridge_cutoff=0.1, sobel_blur=1, min_fiber_size=15,
                   object_properties=None, debug=False):
    """
    """
    channel_xr = data_xr.loc[:, :, :, fiber_channel]
    fov_len = channel_xr.shape[1]

    fiber_label_images = {}
    fiber_object_table = []

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
