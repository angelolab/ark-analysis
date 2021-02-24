import os
import subprocess

import feather
import numpy as np
import pandas as pd
import xarray as xr
import scipy.ndimage as ndimage
from skimage.io import imread

import ark.settings as settings
from ark.utils import io_utils
from ark.utils import misc_utils


def compute_cluster_avg(fovs, channels, base_dir,
                        cluster_dir='pixel_mat_clustered',
                        cluster_avg_name='pixel_cluster_avg.feather'):
    """Averages channel values across all fovs in pixel_mat_clustered

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            Name of the directory to save the pixel files to
        cluster_dir (str):
            Name of the file containing the pixel data with cluster labels
        cluster_avg_name (str):
            Name of file to save the averaged results to
    """

    cluster_avgs = None

    for fov in fovs:
        # read in the fovs data
        fov_pixel_data = feather.read_dataframe(
            os.path.join(base_dir, cluster_dir, fov + '.feather'))

        # aggregate the sums and counts
        sum_by_cluster = fov_pixel_data.groupby('cluster')[channels].sum()
        count_by_cluster = fov_pixel_data.groupby('cluster')[channels].size().to_frame('count')

        # concat the results together
        agg_results = pd.merge(
            sum_by_cluster, count_by_cluster, left_index=True, right_index=True).reset_index()

        # set to cluster_avgs if not defined, otherwise concat
        if cluster_avgs is None:
            cluster_avgs = agg_results
        else:
            cluster_avgs = pd.concat([cluster_avgs, agg_results])

    # sum the counts and the channel sums
    sum_count_totals = cluster_avgs.groupby('cluster')[channels + ['count']].sum().reset_index()

    # now compute the means using the count column
    sum_count_totals[channels] = sum_count_totals[channels].div(sum_count_totals['count'], axis=0)

    # drop the count column
    sum_count_totals = sum_count_totals.drop('count', axis=1)

    # save the DataFrame
    feather.write_dataframe(sum_count_totals,
                            os.path.join(base_dir, cluster_avg_name),
                            compression='uncompressed')


def create_pixel_matrix(img_xr, seg_labels, base_dir,
                        pre_dir='pixel_mat_preprocessed',
                        sub_dir='pixel_mat_subsetted', fovs=None,
                        blur_factor=2, subset_proportion=0.1, seed=42):
    """Preprocess the images for FlowSOM clustering and creates a pixel-level matrix

    Saves preprocessed data to pre_dir and subsetted data to sub_dir

    Args:
        img_xr (xarray.DataArray):
            Array representing image data for each fov
        seg_labels (xarray.DataArray):
            Array representing segmentation labels for each image
        base_dir (str):
            Name of the directory to save the pixel files to
        pre_dir (str):
            Name of the directory which contains the preprocessed pixel data
        sub_dir (str):
            The name of the directory containing the subsetted pixel data
        fovs (list):
            List of fovs to subset over, if None selects all
        blur_factor (int):
            The sigma to set for the Gaussian blur
        subset_proportion (float):
            The proportion of pixels to take from each fov
        seed (int):
            The random seed to set for subsetting
    """

    # if the subset_proportion specified is out of range
    if subset_proportion <= 0 or subset_proportion > 1:
        raise ValueError('Invalid subset percentage entered: must be in (0, 1]')

    # if the base directory doesn't exist
    if not os.path.exists(base_dir):
        raise FileNotFoundError("base_dir %s does not exist" % base_dir)

    # create pre_dir if it doesn't already exist
    if not os.path.exists(os.path.join(base_dir, pre_dir)):
        os.mkdir(os.path.join(base_dir, pre_dir))

    # create sub_dir if it doesn't already exist
    if not os.path.exists(os.path.join(base_dir, sub_dir)):
        os.mkdir(os.path.join(base_dir, sub_dir))

    # set fovs to all if None
    if fovs is None:
        fovs = img_xr.fovs.values

    # verify that the fovs provided are valid
    misc_utils.verify_in_list(fovs=fovs, image_fovs=img_xr.fovs.values)

    # iterate over fovs
    for fov in fovs:
        # subset img_xr with only the fov we're looking for, and cast to float32
        img_data_blur = img_xr.loc[fov, ...].values.astype(np.float32)

        # for each marker, compute the Gaussian blur
        for marker in range(len(img_xr.channels.values)):
            img_data_blur[:, :, marker] = ndimage.gaussian_filter(img_data_blur[:, :, marker],
                                                                  sigma=blur_factor)

        # flatten each image
        pixel_mat = img_data_blur.reshape(-1, len(img_xr.channels.values))

        # convert into a dataframe
        pixel_mat = pd.DataFrame(pixel_mat, columns=img_xr.channels.values)

        # assign metadata about each entry
        pixel_mat['fov'] = fov
        pixel_mat['row_index'] = np.repeat(range(img_data_blur.shape[0]), img_data_blur.shape[1])
        pixel_mat['column_index'] = np.tile(range(img_data_blur.shape[0]), img_data_blur.shape[1])

        # assign segmentation label
        seg_labels_flat = seg_labels.loc[fov, ...].values.flatten()
        pixel_mat['segmentation_label'] = seg_labels_flat

        # needed for selecting just channel columns
        non_num_cols = ['fov', 'row_index', 'column_index', 'segmentation_label']
        chan_cols = [col for col in pixel_mat.columns.values if col not in non_num_cols]

        # remove any rows that sum to 0
        pixel_mat = pixel_mat.loc[pixel_mat.loc[:, chan_cols].sum(axis=1) != 0, :]

        # normalize each row by total marker counts to convert into frequencies
        pixel_mat.loc[:, chan_cols] = pixel_mat.loc[:, chan_cols].div(
            pixel_mat.loc[:, chan_cols].sum(axis=1), axis=0)

        # subset the pixel matrix for training
        pixel_mat_subset = pixel_mat.sample(frac=subset_proportion, random_state=seed)

        # write complete dataset to feather, needed for cluster assignment
        feather.write_dataframe(pixel_mat,
                                os.path.join(base_dir,
                                             pre_dir,
                                             fov + ".feather"),
                                compression='uncompressed')

        # write subseted dataset to hdf5, needed for training
        feather.write_dataframe(pixel_mat_subset,
                                os.path.join(base_dir,
                                             sub_dir,
                                             fov + ".feather"),
                                compression='uncompressed')


def train_som(fovs, channels, base_dir,
              sub_dir='pixel_mat_subsetted', weights_name='weights.feather', num_passes=1):
    """Run the SOM training on the subsetted pixel data.

    Saves weights to base_dir/weights_name.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of markers to subset on
        base_dir (str):
            The path to the data directory
        sub_dir (str):
            The name of the subsetted data directory
        weights_name (str):
            The name of the weights file
        num_passes (int):
            The number of training passes to make through the dataset
    """

    # define the paths to the data
    subsetted_path = os.path.join(base_dir, sub_dir)
    weights_path = os.path.join(base_dir, weights_name)

    # if path to the subsetted file does not exist
    if not os.path.exists(subsetted_path):
        raise FileNotFoundError('Pixel subsetted directory %s does not exist in base_dir %s' %
                                (sub_dir, base_dir))

    # verify that all provided fovs exist in the folder
    files = io_utils.list_files(subsetted_path, substrs='.feather')
    misc_utils.verify_in_list(provided_fovs=fovs,
                              subsetted_fovs=io_utils.remove_file_extensions(files))

    # verify that all the provided channels exist in subsetted data
    sample_sub = feather.read_dataframe(os.path.join(subsetted_path, files[0]))
    misc_utils.verify_in_list(provided_channels=channels,
                              subsetted_channels=sample_sub.columns.values)

    # run the SOM training process
    process_args = ['Rscript', '/create_som_matrix.R', ','.join(fovs), ','.join(channels),
                    str(num_passes), subsetted_path, weights_path]
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


def cluster_pixels(fovs, base_dir, pre_dir='pixel_mat_preprocessed',
                   weights_name='weights.feather', cluster_dir='pixel_mat_clustered'):
    """Uses trained weights to assign cluster labels on full pixel data

    Saves data with cluster labels to cluster_dir

    Args:
        fovs (list):
            The list of fovs to subset on
        base_dir (str):
            The path to the data directory
        pre_dir (str):
            Name of the directory which contains the preprocessed pixel data,
            defaults to pixel_mat_preprocessed
        weights_name (str):
            The name of the weights file
        cluster_dir (str):
            The name of the directory to write the clustered data
    """

    # define the paths to the data
    preprocessed_path = os.path.join(base_dir, pre_dir)
    weights_path = os.path.join(base_dir, weights_name)
    clustered_path = os.path.join(base_dir, cluster_dir)

    # if path to the preprocessed file does not exist
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError('Pixel preprocessed directory %s does not exist in base_dir %s' %
                                (pre_dir, base_dir))

    # if path to the weights file does not exist
    if not os.path.exists(weights_path):
        raise FileNotFoundError('Weights file %s does not exist in base_dir %s' %
                                (weights_name, base_dir))

    # verify that all provided fovs exist in the folder
    files = io_utils.list_files(preprocessed_path, substrs='.feather')
    misc_utils.verify_in_list(provided_fovs=fovs,
                              subsetted_fovs=io_utils.remove_file_extensions(files))

    # ensure the weights columns are valid indexes
    weights = feather.read_dataframe(os.path.join(base_dir, weights_name))
    sample_fov = feather.read_dataframe(os.path.join(base_dir, pre_dir, files[0]))

    misc_utils.verify_in_list(weights_columns=weights.columns.values,
                              pixel_data_columns=sample_fov.columns.values)

    # make the clustered dir if it does not exist
    if not os.path.exists(clustered_path):
        os.mkdir(clustered_path)

    # run the trained SOM on the dataset, assigning clusters
    process_args = ['Rscript', '/run_trained_som.R', ','.join(fovs),
                    preprocessed_path, weights_path, clustered_path]

    process = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # continuously poll the process for output/error so it gets displayed in the Jupyter notebook
    while True:
        # convert from byte string
        output = process.stdout.readline().decode('utf-8')

        # if the output is nothing and the process is done, break
        if process.poll() is not None:
            break
        if output:
            print(output.strip())


def consensus_cluster(fovs, channels, base_dir, max_k=20, cap=3,
                      cluster_dir='pixel_mat_clustered',
                      cluster_avg_name='pixel_cluster_avg.feather',
                      consensus_name='cluster_consensus.feather'):
    """Run consensus clustering algorithm on summed data across channels

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            Name of the directory to save the pixel files to
        max_k (int):
            The number of consensus clusters
        cap (int):
            z-score cap to use when hierarchical clustering
        cluster_dir (str):
            Name of the file containing the pixel data with cluster labels
        cluster_avg_name (str):
            Name of file to save the channel-averaged results to
        consensus_name (str):
            Name of file to save the consensus clustered results
    """

    clustered_path = os.path.join(base_dir, cluster_dir)
    cluster_avg_path = os.path.join(base_dir, cluster_avg_name)
    consensus_path = os.path.join(base_dir, consensus_name)

    if not os.path.exists(clustered_path):
        raise FileNotFoundError('Cluster dir %s does not exist in base_dir %s' %
                                (base_dir, consensus_path))

    # compute and write the averaged cluster results
    compute_cluster_avg(fovs, channels, base_dir, cluster_dir)

    # run the consensus clustering process
    process_args = ['Rscript', '/consensus_cluster.R', ','.join(fovs), ','.join(channels),
                    str(max_k), str(cap), cluster_avg_path, consensus_path]

    process = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # continuously poll the process for output/error so it gets displayed in the Jupyter notebook
    while True:
        # convert from byte string
        output = process.stdout.readline().decode('utf-8')

        # if the output is nothing and the process is done, break
        if process.poll() is not None:
            break
        if output:
            print(output.strip())
