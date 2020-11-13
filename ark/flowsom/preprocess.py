import os

import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndimage
from skimage.io import imread

import ark.settings as settings
from ark.utils import load_utils
from ark.utils import misc_utils


def preprocess_flowsom(img_xr, seg_labels, save_path,
                       fovs=None, channels=None, blur_factor=2):
    """Preprocess the images for FlowSOM clustering

    Args:
        img_data (xarray.DataArray):
            Array representing image data for each fov
        seg_labels (xarray.DataArray):
            Array representing segmentation labels for each image
        save_dir (str):
            The directory to write the pixel cluster CSVs to
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
    misc_utils.verify_in_list(fovs=fovs, image_fovs=img_xr.coords['fovs'].values)
    misc_utils.verify_in_list(channels=channels, image_channels=img_xr.coords['channels'].values)

    # verify the save path exists
    if not os.path.exists(save_path):
        raise ValueError("save_path provided does not exist")

    img_xr_subset = img_xr.loc[fovs, :, :, channels]

    # create an xarray to store the pre-processed data, assuming square images
    # flat_size = img_xr.shape[1]**2
    # img_xr_proc = xr.DataArray(np.zeros((len(fovs), flat_size, len(channels))),
    #                            coords=[fovs, np.arange(flat_size), channels],
    #                            dims=["fovs", "pixel_num", "channels"])

    # iterate over fovs
    for fov in fovs:
        # subset img_xr with only the fov we're looking for
        img_xr_sub = img_xr_subset.loc[fov, ...].values

        # apply a Gaussian blur for each marker
        img_data_blur = np.apply_along_axis(ndimage.gaussian_filter, axis=2,
                                            arr=img_xr_sub, sigma=blur_factor)

        # flatten each image, assuming each image is square for now
        pixel_mat = np.reshape(img_data_blur, (img_data_blur.shape[0]**2, len(channels)))

        # NOTE: the following lines may be unnecessary, this is assuming that
        # people who use FlowSOM expect pixel cluster matrices to be referenced

        # assign the fov to each cell
        fov_labels = np.repeat(int(fov.replace('Point', '')), img_data_blur.shape[0]**2)
        pixel_mat = np.concatenate([pixel_mat, np.reshape(fov_labels, (-1, 1))], axis=1)

        # assign x and y coords
        x_coords = np.repeat(range(img_data_blur.shape[0]), img_data_blur.shape[0])
        pixel_mat = np.concatenate([pixel_mat, np.reshape(x_coords, (-1, 1))], axis=1)

        y_coords = np.tile(range(img_data_blur.shape[0]), img_data_blur.shape[0])
        pixel_mat = np.concatenate([pixel_mat, np.reshape(y_coords, (-1, 1))], axis=1)

        # assign the corresponding segmentation labels
        seg_labels_flat = seg_labels.loc[fov, ...].values.flatten()
        pixel_mat = np.concatenate([pixel_mat, np.reshape(seg_labels_flat, (-1, 1))], axis=1)

        # remove zero pixels
        pixel_mat_nonzero = pixel_mat[np.sum(pixel_mat[:, :len(channels)], axis=1) != 0, :]

        header = ','.join(channels) + ',sample,x,y,label'
        np.savetxt(os.path.join(save_path, fov + '_pixel_info.csv'), pixel_mat_nonzero,
                   delimiter=',', header=header)

        # # assign to respective fov in img_xr_proc
        # img_xr_proc.loc[fov, ...] = img_data_flat

    # return img_xr_proc
