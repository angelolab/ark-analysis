import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt

from skimage.filters import sobel, threshold_multiotsu, meijering
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import remove_small_objects, erosion, dilation
from skimage.measure import regionprops
from skimage.exposure import equalize_adapthist


def segment_fibers(data_xr, fiber_channel, out_dir, blur=2, contrast_scaling_divisor=128,
                   fiber_widths=(2, 4, 6), ridge_cutoff=0.1, sobel_blur=1, min_fiber_size=15):
    """
    """
    channel_xr = data_xr.loc[:, :, :, fiber_channel]
    fov_len = channel_xr.shape[1]
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
