import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.segmentation import find_boundaries
from skimage.exposure import rescale_intensity

from ark.utils import misc_utils

# plotting functions
from ark.utils.misc_utils import verify_in_list


def plot_clustering_result(img_xr, fovs, save_dir=None, cmap='tab20',
                           fov_col='fovs', figsize=(10, 10)):
    """Takes an xarray containing labeled images and displays them.

    Args:
        img_xr (xr.DataArray):
            xarray containing labeled cell objects.
        fovs (list):
            list of fovs to display.
        save_dir (str):
            If provided, the image will be saved to this location.
        cmap (str):
            Cmap to use for the image that will be displayed.
        fov_col (str):
            column with the fovs names in img_xr.
        figsize (tuple):
            Size of the image that will be displayed.
    """

    verify_in_list(fov_names=fovs, unique_fovs=img_xr.fovs)

    for fov in fovs:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        plt.title(fov)
        plt.imshow(img_xr[img_xr[fov_col] == fov].values.squeeze(), cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
        if save_dir:
            misc_utils.save_figure(save_dir, f'{fov}.png')


def tif_overlay_preprocess(segmentation_labels, plotting_tif):
    """Validates plotting_tif and preprocesses it accordingly

    Args:
        segmentation_labels (numpy.ndarray):
            2D numpy array of labeled cell objects
        plotting_tif (numpy.ndarray):
            2D or 3D numpy array of imaging signal

    Returns:
        numpy.ndarray:
            The preprocessed image
    """

    if len(plotting_tif.shape) == 2:
        if plotting_tif.shape != segmentation_labels.shape:
            raise ValueError("plotting_tif and segmentation_labels array dimensions not equal.")
        else:
            # convert RGB image with the blue channel containing the plotting tif data
            formatted_tif = np.zeros((plotting_tif.shape[0], plotting_tif.shape[1], 3),
                                     dtype=plotting_tif.dtype)
            formatted_tif[..., 2] = plotting_tif
    elif len(plotting_tif.shape) == 3:
        # can only support up to 3 channels
        if plotting_tif.shape[2] > 3:
            raise ValueError("max 3 channels of overlay supported, got {}".
                             format(plotting_tif.shape))

        # set first n channels (in reverse order) of formatted_tif to plotting_tif
        # (n = num channels in plotting_tif)
        formatted_tif = np.zeros((plotting_tif.shape[0], plotting_tif.shape[1], 3),
                                 dtype=plotting_tif.dtype)
        formatted_tif[..., :plotting_tif.shape[2]] = plotting_tif
        formatted_tif = np.flip(formatted_tif, axis=2)
    else:
        raise ValueError("plotting tif must be 2D or 3D array, got {}".
                         format(plotting_tif.shape))

    return formatted_tif


def create_overlay(segmentation_labels, plotting_tif, alternate_segmentation=None):
    """Take in labeled contour data, along with optional mibi tif and second contour,
    and overlay them for comparison"

    Generates the outline(s) of the mask(s) as well as intensity from plotting tif. Predicted
    contours are colored red, while alternate contours are colored white.

    Args:
        segmentation_labels (numpy.ndarray):
            2D numpy array of labeled cell objects
        plotting_tif (numpy.ndarray):
            2D or 3D numpy array of imaging signal
        alternate_segmentation (numpy.ndarray):
            2D numpy array of labeled cell objects

    Returns:
        numpy.ndarray:
            The image with the channel overlay
    """

    plotting_tif = tif_overlay_preprocess(segmentation_labels, plotting_tif)

    # define borders of cells in mask
    predicted_contour_mask = find_boundaries(segmentation_labels,
                                             connectivity=1, mode='inner').astype(np.uint8)
    predicted_contour_mask[predicted_contour_mask > 0] = 255

    # rescale each channel to go from 0 to 255
    rescaled = np.zeros(plotting_tif.shape, dtype='uint8')

    for idx in range(plotting_tif.shape[2]):
        if np.max(plotting_tif[:, :, idx]) == 0:
            # don't need to rescale this channel
            pass
        else:
            percentiles = np.percentile(plotting_tif[:, :, idx][plotting_tif[:, :, idx] > 0],
                                        [5, 95])
            rescaled_intensity = rescale_intensity(plotting_tif[:, :, idx],
                                                   in_range=(percentiles[0], percentiles[1]),
                                                   out_range='uint8')
            rescaled[:, :, idx] = rescaled_intensity

    # overlay first contour on all three RGB, to have it show up as white border
    rescaled[predicted_contour_mask > 0, :] = 255

    # overlay second contour as red outline if present
    if alternate_segmentation is not None:

        if segmentation_labels.shape != alternate_segmentation.shape:
            raise ValueError("segmentation_labels and alternate_"
                             "segmentation array dimensions not equal.")

        # define borders of cell in mask
        alternate_contour_mask = find_boundaries(alternate_segmentation, connectivity=1,
                                                 mode='inner').astype(np.uint8)
        rescaled[alternate_contour_mask > 0, 0] = 255
        rescaled[alternate_contour_mask > 0, 1:] = 0

    return rescaled
