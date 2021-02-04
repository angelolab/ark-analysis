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
                        pre_name='pixel_mat_preprocessed.hdf5',
                        sub_name='pixel_mat_subsetted.hdf5', fovs=None, channels=None,
                        blur_factor=2, subset_percent=0.1):
    """Preprocess the images for FlowSOM clustering and creates a pixel-level matrix

    Args:
        img_xr (xarray.DataArray):
            Array representing image data for each fov
        seg_labels (xarray.DataArray):
            Array representing segmentation labels for each image
        base_dir (str):
            Name of the directory to save the pixel files to
        pre_name (str):
            Name of the file which contains the preprocessed pixel data,
            defaults to pixel_mat_preprocessed.hdf5
        sub_name (str):
            The name of the subsetted file name, defaults to pixel_mat_subsetted.hdf5
        fovs (list):
            List of fovs to subset over, if None selects all
        channels (list):
            List of channels to subset over, if None selects all
        blur_factor (int):
            The sigma to set for the Gaussian blur
        subset_percent (float):
            The percentage of pixels to take from each fov, defaults to 0.1
    """

    if subset_percent <= 0 or subset_percent > 1:
        raise ValueError('Invalid subset percentage entered: must be in (0, 1]')

    if not os.path.exists(base_dir):
        raise FileNotFoundError("Path to base_dir %s does not exist" % base_dir)

    # set fovs to all if None
    if fovs is None:
        fovs = img_xr.fovs.values

    # set channels to all if None
    if channels is None:
        channels = img_xr.channels.values

    # verify that the fovs and channels provided are valid
    misc_utils.verify_in_list(fovs=fovs, image_fovs=img_xr.fovs.values)
    misc_utils.verify_in_list(channels=channels, image_channels=img_xr.channels.values)

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

        # subset the pixel matrix for training
        pixel_mat_subset = pixel_mat.sample(frac=subset_percent)

        # write complete dataset to hdf5, needed for cluster assignment
        pixel_mat.to_hdf(os.path.join(base_dir, pre_name), key=fov, mode='a',
                         format='table', data_columns=True)

        # write subseted dataset to hdf5, needed for training
        pixel_mat_subset.to_hdf(os.path.join(base_dir, sub_name), key=fov, mode='a',
                                format='table', data_columns=True)


def train_som(fovs, channels, base_dir,
              subset_name='pixel_mat_subsetted.hdf5', weights_name='weights.hdf5'):
    """Run the SOM training on the subsetted pixel data.

    Saves weights to base_dir/weights_name.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of markers to subset on
        base_dir (str):
            The path to the data directory
        subset_name (str):
            The name of the subsetted file name, defaults to pixel_mat_subsetted.hdf5
        weights_name (str):
            The name of the weights file, defaults to weights.hdf5
    """

    subsetted_path = os.path.join(base_dir, subset_name)
    weights_path = os.path.join(base_dir, weights_name)

    # if path to the subsetted file does not exist
    if not os.path.exists(subsetted_path):
        raise FileNotFoundError('Pixel subsetted HDF5 %s does not exist in base_dir %s' %
                                (subset_name, base_dir))

    # run som_train.R
    process_args = ['Rscript', '/som_train.R', ','.join(fovs), ','.join(channels),
                    subsetted_path, weights_path]
    process = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # continuously poll the process for output/error to display in Jupyter notebook
    while True:
        # convert from byte string
        output = process.stdout.readline().decode('utf-8')

        # if the output is nothing and the process is done, break
        if process.poll() is not None:
            break
        if output:
            print(output.strip())


def cluster_pixels(fovs, channels, base_dir, pre_name='pixel_mat_preprocessed.hdf5',
                   weights_name='weights.hdf5', cluster_name='pixel_mat_clustered.hdf5'):
    """Uses trained weights to assign cluster labels on full pixel data

    Saves data with cluster labels to base_dir/pixel_mat_clustered.hdf5

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of markers to subset on
        base_dir (str):
            The path to the data directory
        pre_name (str):
            The name of the preprocessed file name, defaults to pixel_mat_preprocessed.hdf5
        weights_name (str):
            The name of the weights file, defaults to weights.hdf5
        cluster_name (str):
            The name of the file to write the clustered csv to, default to pixel_mat_clustered.hdf5
    """

    preprocessed_path = os.path.join(base_dir, pre_name)
    weights_path = os.path.join(base_dir, weights_name)
    clustered_path = os.path.join(base_dir, cluster_name)

    # if path to the preprocessed file does not exist
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError('Pixel preprocessed HDF5 %s does not exist in base_dir %s' %
                                (pre_name, base_dir))

    # if path to the weights file does not exist
    if not os.path.exists(weights_path):
        raise FileNotFoundError('Weights HDF5 %s does not exist in base_dir %s' %
                                (weights_name, base_dir))

    process_args = ['Rscript', '/som_cluster.R', ','.join(fovs), ','.join(channels),
                    preprocessed_path, weights_path, clustered_path]

    process = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # continuously poll the process for output/error so it gets displayed in the Jupyter notebook
    while True:
        # convert from byte string
        output = process.stdout.readline().decode('utf-8')

        # if the output is nothing and the process is done, break
        if process.poll() is not None:
            break
        if output:
            print(output.strip())


# def cluster_pixels(fovs, channels, base_dir,
#                    pre_name='pixel_mat_preprocessed.hdf5', subset_name='pixel_mat_subsetted.hdf5',
#                    weights_name='weights.hdf5', cluster_name='pixel_mat_clustered.hdf5'):
#     """Run the FlowSOM training on the pixel data.

#     Saves results to pixel_mat_clustered.csv in base_dir.

#     Args:
#         fovs (list):
#             The list of fovs to subset on
#         channels (list):
#             The list of markers to subset on
#         base_dir (str):
#             The path to the data directory
#         pre_name (str):
#             The name of the preprocessed file name, defaults to pixel_mat_preprocessed.hdf5
#         subset_name (str):
#             The name of the subsetted file name, defaults to pixel_mat_subsetted.hdf5
#         weights_name (str):
#             The name of the weights file, defaults to weights.hdf5
#         cluster_name (str):
#             The name of the file to write the clustered csv to, default to pixel_mat_clustered.hdf5
#     """

#     # set the paths to the preprocessed matrix and clustered matrix
#     preprocessed_path = os.path.join(base_dir, pre_name)
#     subsetted_path = os.path.join(base_dir, subset_name)
#     weights_path = os.path.join(base_dir, weights_name)
#     clustered_path = os.path.join(base_dir, cluster_name)

#     # if path to the preprocessed file does not exist
#     if not os.path.exists(preprocessed_path):
#         raise FileNotFoundError('Pixel preprocessed HDF5 %s does not exist in base_dir %s' %
#                                 (pixel_pre_name, base_dir))

#     # if path to the subsetted file does not exist
#     if not os.path.exists(subsetted_path):
#         raise FileNotFoundError('Pixel subsetted CSV %s does not exist in base_dir %s' %
#                                 (pixel_subset_name, base_dir))

#     # use Rscript to run som_runner.R with the correct command line args
#     process_args = ['Rscript', '/som_runner.R', ','.join(fovs), ','.join(channels),
#                     preprocessed_path, subsetted_path, weights_path, clustered_path]
#     process = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     # continuously poll the process for output/error so it gets displayed in the Jupyter notebook
#     while True:
#         # handle stdout, convert from byte string
#         output = process.stdout.readline().decode('utf-8')

#         # if the output is nothing and the process is done, break
#         if process.poll() is not None:
#             break
#         if output:
#             print(output.strip())

#         # handle error, convert from byte string
#         error = process.stderr.readline().decode('utf-8')

#         # print the error, includes non-error stuff so don't break (allow output check to handle)
#         if error:
#             print(error.strip())
