import os
import copy

import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage.segmentation import find_boundaries
from skimage.exposure import rescale_intensity


def plot_overlay(predicted_contour, plotting_tif, alternate_contour=None, path=None, show=False):
    """Take in labeled contour data, along with optional mibi tif and second contour,
    and overlay them for comparison"

    Plots the outline(s) of the mask(s) as well as intensity from plotting tif.  Predicted
    contours are plotted in red, while alternate contours are plotted in white.  Plot is
    saved as a TIF file in provided file path, if specified.

    Args:
        predicted_contour (numpy.ndarray):
            2D numpy array of labeled cell objects
        plotting_tif (numpy.ndarray):
            2D or 3D numpy array of imaging signal
        alternate_contour (numpy.ndarray):
            2D numpy array of labeled cell objects
        path (str):
            path to save the resulting image
        show (bool):
            whether or not to show plot
    """

    if plotting_tif is None:
        # will just plot the outlines
        pass
    else:
        if len(plotting_tif.shape) == 2:
            if plotting_tif.shape != predicted_contour.shape:
                raise ValueError("plotting_tif and predicted_contour array dimensions not equal.")
            else:
                # convert RGB image with same data across all three channels
                plotting_tif = np.stack((plotting_tif, plotting_tif, plotting_tif), axis=2)
        elif len(plotting_tif.shape) == 3:
            blank_channel = np.zeros(plotting_tif.shape[:2] + (1,), dtype=plotting_tif.dtype)
            if plotting_tif.shape[2] == 1:
                # pad two empty channels
                plotting_tif = np.concatenate((plotting_tif, blank_channel, blank_channel), axis=2)
            elif plotting_tif.shape[2] == 2:
                # pad one empty channel
                plotting_tif = np.concatenate((plotting_tif, blank_channel), axis=2)
            elif plotting_tif.shape[2] == 3:
                # don't need to do anything
                pass
            else:
                raise ValueError("only 3 channels of overlay supported, got {}".
                                 format(plotting_tif.shape))
        else:
            raise ValueError("plotting tif must be 2D or 3D array, got {}".
                             format(plotting_tif.shape))

    if path is not None:
        if os.path.exists(os.path.split(path)[0]) is False:
            raise ValueError("File path does not exist.")

    # define borders of cells in mask
    predicted_contour_mask = find_boundaries(predicted_contour,
                                             connectivity=1, mode='inner').astype(np.uint8)
    predicted_contour_mask[predicted_contour_mask > 0] = 255

    if plotting_tif is None:
        # will just save the contour mask
        io.imsave(path, predicted_contour_mask)
    else:
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

        # save as TIF if path supplied, otherwise display on screen
        if path is not None:
            io.imsave(path, rescaled)
        if show:
            io.imshow(rescaled)
