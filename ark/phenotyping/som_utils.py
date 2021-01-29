import multiprocessing
import os
import subprocess

import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndimage
from skimage.io import imread

import ark.settings as settings
from ark.utils import load_utils
from ark.utils import misc_utils


def create_pixel_matrix(img_xr, seg_labels, base_dir,
                        hdf_name='pixel_mat_preprocessed.hdf5', fovs=None,
                        channels=None, blur_factor=2):
    """Preprocess the images for FlowSOM clustering and creates a pixel-level matrix

    Args:
        img_xr (xarray.DataArray):
            Array representing image data for each fov
        seg_labels (xarray.DataArray):
            Array representing segmentation labels for each image
        base_dir (str):
            Name of the directory to save the pixel files to
        hdf_name (str):
            Name of the file to save the pixel files to, defaults to pixel_mat_preprocessed.hdf5
        fovs (list):
            List of fovs to subset over, if None selects all
        channels (list):
            List of channels to subset over, if None selects all
        blur_factor (int):
            The sigma to set for the Gaussian blur
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

    # verify that the path to base_dir is valid
    if not os.path.exists(base_dir):
        raise FileNotFoundError('Path to base_dir %s does not exist' % base_dir)

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

        # write dataset to hdf5 file, mode 'a' will create if it doesn't exist
        pixel_mat.to_hdf(os.path.join(base_dir, hdf_name), key=fov, mode='a')


def subset_pixels(fovs, base_dir, hdf_name='pixel_mat_preprocessed.hdf5',
                  csv_name='pixel_mat_subsetted.csv', subset_percent=0.1):
    """Takes a random percentage subset of pixel data from each fov

    Args:
        fovs (list):
            The list of fovs to read
        base_dir (str):
            Name of the directory to save the subsetted CSV to
        hdf_name (str):
            Name of the file which contains the preprocessed pixel data,
            defaults to pixel_mat_preprocessed.hdf5
        csv_name (str):
            Name of the file to write the subsetted pixel data frame to
        subset_percent (float):
            The percentage of pixels to take from each fov, defaults to 0.1
    """

    # ensure subset percent is valid
    if subset_percent <= 0 or subset_percent > 1:
        raise ValueError('Subset percent provided must be in (0, 1]')

    # ensure the file path to hdf_name exists
    if not os.path.exists(os.path.join(base_dir, hdf_name)):
        raise FileNotFoundError('Preprocessed HDF5 %s not found in base_dir %s' %
                                (hdf_name, base_dir))

    # define our subsetted flowsom matrix
    flowsom_subset_data = None

    for fov in fovs:
        # read the specific fov key from the HDF5
        fov_pixel_data = pd.read_hdf(os.path.join(base_dir, hdf_name), key=fov)

        # subset the data per fov using the subset_percent argument
        fov_pixel_data = fov_pixel_data.sample(frac=subset_percent)

        # assign to flowsom_subset_data if not already assigned, otherwise concatenates
        if flowsom_subset_data is None:
            flowsom_subset_data = fov_pixel_data
        else:
            flowsom_subset_data = pd.concat([flowsom_subset_data, fov_pixel_data])

    flowsom_subset_data.to_csv(os.path.join(base_dir, csv_name), index=False)


def cluster_pixels(fovs, channels, base_dir,
                   pixel_pre_name='pixel_mat_preprocessed.hdf5',
                   pixel_subset_name='pixel_mat_subsetted.csv',
                   pixel_cluster_name='pixel_mat_clustered.hdf5'):
    """Run the FlowSOM training on the pixel data.

    Saves results to pixel_mat_clustered.csv in base_dir.
    Usage: Rscript som_runner.R {fovs} {chan_list} {pixel_matrix_path}
    {pixel_subset_path} {save_path}

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of markers to subset on
        base_dir (str):
            The path to the data directory
        pixel_pre_name (str):
            The name of the preprocessed file name, defaults to pixel_mat_preprocessed.csv
        pixel_subset_name (str):
            The name of the subsetted file name, defaults to pixel_mat_subsetted.csv
        pixel_cluster_name (str):
            The name of the file to write the clustered csv to, default to pixel_mat_clustered.hdf5
    """

    # set the paths to the preprocessed matrix and clustered matrix
    preprocessed_path = os.path.join(base_dir, pixel_pre_name)
    subsetted_path = os.path.join(base_dir, pixel_subset_name)
    clustered_path = os.path.join(base_dir, pixel_cluster_name)

    # if path to the preprocessed file does not exist
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError('Pixel preprocessed HDF5 %s does not exist in base_dir %s' %
                                (pixel_pre_name, base_dir))

    # if path to the subsetted file does not exist
    if not os.path.exists(subsetted_path):
        raise FileNotFoundError('Pixel subsetted CSV %s does not exist in base_dir %s' %
                                (pixel_subset_name, base_dir))

    # use Rscript to run som_runner.R with the correct command line args
    subprocess.call(['Rscript', '/som_runner.R', ','.join(fovs), ','.join(chan_list),
                     preprocessed_path, subsetted_path, clustered_path])
