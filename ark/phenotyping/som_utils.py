import os
import multiprocessing
import subprocess

import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndimage
from skimage.io import imread

import ark.settings as settings
from ark.utils import load_utils
from ark.utils import misc_utils


def create_pixel_matrix(img_xr, seg_labels, fovs=None, channels=None, blur_factor=2):
    """Preprocess the images for FlowSOM clustering and creates a pixel-level matrix

    Args:
        img_xr (xarray.DataArray):
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
        fovs = img_xr.fovs.values

    # set channels to all if None
    if channels is None:
        channels = img_xr.channels.values

    # verify that the fovs and channels provided are valid
    misc_utils.verify_in_list(fovs=fovs, image_fovs=img_xr.fovs.values)
    misc_utils.verify_in_list(channels=channels, image_channels=img_xr.channels.values)

    # define our flowsom matrix
    flowsom_data = None

    # iterate over fovs
    for fov in fovs:
        # subset img_xr with only the fov we're looking for, and cast to float32
        img_data_blur = img_xr.loc[fov, ..., channels].values.astype(np.float32)

        # for each marker, compute the Gaussian blur
        for marker in range(len(channels)):
            img_data_blur[:, :, marker] = ndimage.gaussian_filter(img_data_blur[:, :, marker],
                                                                  sigma=blur_factor)

        # flatten each image
        pixel_mat = img_data_blur.reshape(-1, len(channels))

        # convert into a dataframe
        pixel_mat = pd.DataFrame(pixel_mat, columns=channels)

        # assign metadata about each entry
        pixel_mat['fov'] = fov
        pixel_mat['row_index'] = np.repeat(range(img_data_blur.shape[0]), img_data_blur.shape[1])
        pixel_mat['column_index'] = np.tile(range(img_data_blur.shape[0]), img_data_blur.shape[1])

        # assign segmentation label
        seg_labels_flat = seg_labels.loc[fov, ...].values.flatten()
        pixel_mat['segmentation_label'] = seg_labels_flat

        # remove any rows that sum to 0
        pixel_mat = pixel_mat.loc[pixel_mat.loc[:, channels].sum(axis=1) != 0, :]

        # normalize each row by total marker counts to convert into frequencies
        pixel_mat.loc[:, channels] = pixel_mat.loc[:, channels].div(
            pixel_mat.loc[:, channels].sum(axis=1), axis=0)

        # assign to flowsom_data if not already assigned, otherwise concatenates
        if flowsom_data is None:
            flowsom_data = pixel_mat
        else:
            flowsom_data = pd.concat([flowsom_data, pixel_mat])

    # normalize each marker column by the 99.9 percentile value
    flowsom_data.loc[:, channels] = flowsom_data.loc[:, channels].div(
        flowsom_data.loc[:, channels].quantile(q=0.999, axis=0), axis=1)

    return flowsom_data


def cluster_pixels(base_dir, chan_list):
    """Run the FlowSOM training on the pixel data

    Usage: Rscript som_runner.R {path_to_pixel_matrix} {chan_list_comma_separated} {save_path}

    Args:
        base_dir (str):
            The path to the base directory
        chan_list (list):
            The list of markers to subset on
    """

    # path_to_som_runner = os.path.join(os.path.dirname(os.path.realpath(__file__),
    #                                   '..', 'som_runner.R'))

    print(os.istdir('.'))
    print(os.path.join(os.path.dirname(os.path.realpath(__file__))))
    print(os.listdir(os.path.dirname(os.path.realpath(__file__))))

    subprocess.call(['Rscript', 'som_runner.R',
                     os.path.join(base_dir, 'example_pixel_matrix.csv'),
                     ','.join(chan_list), base_dir])
