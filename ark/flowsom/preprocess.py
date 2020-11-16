import os

import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndimage
from skimage.io import imread

import ark.settings as settings
from ark.utils import load_utils
from ark.utils import misc_utils


def preprocess_flowsom(img_xr, seg_labels, fovs=None, channels=None, blur_factor=2):
    """Preprocess the images for FlowSOM clustering and writes to file

    Args:
        img_data (xarray.DataArray):
            Array representing image data for each fov
        seg_labels (xarray.DataArray):
            Array representing segmentation labels for each image
        fovs (list):
            List of fovs to subset over, if None selects all
        channels (list):
            List of channels to subset over, if None selects all
        blur_factor (int):
            The sigma to set for the Gaussian blur

    Returns:
        pandas.DataFrame:
            A matrix with pixel-level channel information for non-zero pixels in img_xr
    """

    # set fovs to all if None
    if fovs is None:
        fovs = img_xr.coords['fovs'].values

    # set channels to all if None
    if channels is None:
        channels = img_xr.coords['channels'].values

    # verify that the fovs and channels provided are valid
    misc_utils.verify_in_list(fovs=fovs, image_fovs=img_xr.coords['fovs'].values)
    misc_utils.verify_in_list(channels=channels, image_channels=img_xr.coords['channels'].values)

    # delete any fovs and channels we don't need
    # NOTE: no idea why Candace didn't use xarrays
    img_xr_subset = img_xr.loc[fovs, :, :, channels]

    # define our flowsom matrix
    flowsom_data = None

    # iterate over fovs
    for fov in fovs:
        # subset img_xr with only the fov we're looking for
        img_xr_sub = img_xr_subset.loc[fov, ...].values

        # apply a Gaussian blur for each marker
        img_data_blur = np.apply_along_axis(ndimage.gaussian_filter, axis=2,
                                            arr=img_xr_sub, sigma=blur_factor)

        # flatten each image, assuming each image is square for now
        pixel_mat = np.reshape(img_data_blur, (img_data_blur.shape[0]**2, len(channels)))

        # convert into a dataframe
        pixel_mat = pd.DataFrame(pixel_mat, columns=channels)

        # assign metadata about each entry
        # NOTE: no idea why Candace didn't use pandas
        pixel_mat['fov'] = fov
        pixel_mat['x_coord'] = np.repeat(range(img_data_blur.shape[0]), img_data_blur.shape[0])
        pixel_mat['y_coord'] = np.tile(range(img_data_blur.shape[0]), img_data_blur.shape[0])

        # assign segmentation label
        seg_labels_flat = seg_labels.loc[fov, ...].values.flatten()
        pixel_mat['seg_label'] = seg_labels_flat

        # remove any rows that sum to 0
        pixel_mat = pixel_mat.loc[pixel_mat.loc[:, channels].sum(axis=1) != 0, :]

        # NOTE: no idea why Candace didn't do these preprocessing steps in Python
        # turn into frequency
        pixel_mat.loc[:, channels] = pixel_mat.loc[:, channels].div(
            pixel_mat.sum(axis=1), axis=0)

        # 99.9% normalization
        pixel_mat.loc[:, channels] = pixel_mat.loc[:, channels].div(
            pixel_mat.quantile(q=0.999, axis=1), axis=0)

        # assign to flowsom_data if not already assigned, otherwise concatenates
        if flowsom_data is None:
            flowsom_data = pixel_mat
        else:
            flowsom_data = pd.concat([flowsom_data, pixel_mat])

    return flowsom_data
