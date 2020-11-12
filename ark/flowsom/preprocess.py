import os

import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from skimage.io import imread

import ark.settings as settings
from ark.utils import load_utils
from ark.utils import misc_utils


def preprocess_flowsom(img_xr, fovs=None, channels=None, blur_factor=2):
    """Preprocess the images for FlowSOM clustering

    Args:
        img_data (xarray.DataArray):
            Array representing image data for each fov
        fovs (list):
            List of fovs to subset over, if None selects all
        channels (list):
            List of channels to subset over, if None selects all
        blur_factor (int):
            The sigma to set for the Gaussian blur

    Returns:
        xarray.DataArray:
            Array containing the Gaussian blurred and flattened image data
    """

    # set fovs to all if None
    if fovs is None:
        fovs = img_xr.coords['fovs'].values

    # set channels to all if None
    if channels is None:
        channels = img_xr.coords['channels'].values

    # verify that the fovs and channels provided are valid
    misc_utils.verify_in_list(fovs, img_xr.coords['fovs'].values)
    misc_utils.verify_in_list(channels, img_xr.coords['fovs'].values)

    img_xr_subset = img_xr.loc[fovs, :, :, channels]

    # create an xarray to store the pre-processed data, assuming square images
    flat_size = img_xr.shape[1]**2
    img_xr_proc = xr.DataArray(np.zeros(len(fovs), flat_size, len(channels)),
                               coords=[fovs, np.arange(flat_size), channels],
                               dims=["fovs", "pixel_num", "channels"])

    # iterate over fovs
    for fov in fovs:
        # subset img_xr with only the fov we're looking for
        img_xr_sub = img_xr.loc[fov, ...].values

        # apply a Gaussian blur for each marker
        img_data_blur = np.apply_along_axis(ndimage.gaussian_filter, axis=2,
                                            arr=img_xr_sub, sigma=blur_factor)

        # flatten each image, assuming each image is square for now
        img_data_flat = np.reshape(img_data_blur.shape[0]**2, len(channels))

        # assign to respective fov in img_xr_proc
        img_xr_proc.loc[fov, ...] = img_data_flat

    return img_xr_proc
