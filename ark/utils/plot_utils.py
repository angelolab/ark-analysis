import os
import copy

import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage.segmentation import find_boundaries
from skimage.exposure import rescale_intensity


def preprocess_tif(predicted_contour, plotting_tif):
    """Validates plotting_tif and preprocesses it accordingly

    Args:
        predicted_contour (numpy.ndarray):
            2D numpy array of labeled cell objects
        plotting_tif (numpy.ndarray):
            2D or 3D numpy array of imaging signal

    Returns:
        numpy.ndarray:
            The preprocessed image
    """

    if len(plotting_tif.shape) == 2:
        if plotting_tif.shape != predicted_contour.shape:
            raise ValueError("plotting_tif and predicted_contour array dimensions not equal.")
        else:
            # convert RGB image with same data across all three channels
            formatted_tif = np.stack((plotting_tif, plotting_tif, plotting_tif), axis=2)
    elif len(plotting_tif.shape) == 3:
        # can only support up to 3 channels
        if plotting_tif.shape[2] > 3:
            raise ValueError("max 3 channels of overlay supported, got {}".
                             format(plotting_tif.shape))

        # set first n channels of formatted_tif to plotting_tif (n = num channels in plotting_tif)
        formatted_tif = np.zeros((plotting_tif.shape[0], plotting_tif.shape[1], 3),
                                 dtype=plotting_tif.dtype)
        formatted_tif[..., :plotting_tif.shape[2]] = plotting_tif
    else:
        raise ValueError("plotting tif must be 2D or 3D array, got {}".
                         format(plotting_tif.shape))

    return formatted_tif


def create_overlay(predicted_contour, plotting_tif, alternate_contour=None):
    """Take in labeled contour data, along with optional mibi tif and second contour,
    and overlay them for comparison"

    Generates the outline(s) of the mask(s) as well as intensity from plotting tif. Predicted
    contours are colored red red, while alternate contours are colored white.

    Args:
        predicted_contour (numpy.ndarray):
            2D numpy array of labeled cell objects
        plotting_tif (numpy.ndarray):
            2D or 3D numpy array of imaging signal
        alternate_contour (numpy.ndarray):
            2D numpy array of labeled cell objects
    """

    plotting_tif = preprocess_tif(predicted_contour, plotting_tif)

    # define borders of cells in mask
    predicted_contour_mask = find_boundaries(predicted_contour,
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
    if alternate_contour is not None:

        if predicted_contour.shape != alternate_contour.shape:
            raise ValueError("predicted_contour and alternate_"
                             "contour array dimensions not equal.")

        # define borders of cell in mask
        alternate_contour_mask = find_boundaries(alternate_contour, connectivity=1,
                                                 mode='inner').astype(np.uint8)
        rescaled[alternate_contour_mask > 0, 0] = 255
        rescaled[alternate_contour_mask > 0, 1:] = 0

    return rescaled
