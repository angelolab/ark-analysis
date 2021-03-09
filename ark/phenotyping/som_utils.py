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
from ark.utils import load_utils
from ark.utils import misc_utils


def compute_cluster_avg(fovs, channels, base_dir,
                        cluster_dir='pixel_mat_clustered',
                        cluster_avg_name='pixel_cluster_avg.feather',):
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

    # define the cluster averages DataFrame
    cluster_avgs = pd.DataFrame()

    for fov in fovs:
        # read in the fovs data
        fov_pixel_data = feather.read_dataframe(
            os.path.join(base_dir, cluster_dir, fov + '.feather')
        )

        # aggregate the sums and counts
        sum_by_cluster = fov_pixel_data.groupby('cluster')[channels].sum()
        count_by_cluster = fov_pixel_data.groupby('cluster')[channels].size().to_frame('count')

        # concat the results together
        agg_results = pd.merge(
            sum_by_cluster, count_by_cluster, left_index=True, right_index=True).reset_index()

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


def create_fov_pixel_data(fov, channels, img_data, seg_labels,
                          blur_factor=2, subset_proportion=0.1, seed=42):
    """Preprocess pixel data for one fov

    Saves preprocessed data to pre_dir and subsetted data to sub_dir

    Args:
        fov (str):
            Name of the fov to index
        channels (list):
            List of channels to subset over
        img_data (numpy.ndarray):
            Array representing image data for one fov
        seg_labels (numpy.ndarray):
            Array representing segmentation labels for one fov
        blur_factor (int):
            The sigma to set for the Gaussian blur
        subset_proportion (float):
            The proportion of pixels to take from each fov
        seed (int):
            The random seed to set for subsetting

    Returns:
        tuple:
            A tuple containing two pd.Dataframes:

            - The full preprocessed pixel dataset for a fov
            - The subsetted pixel dataset for a fov
    """

    # for each marker, compute the Gaussian blur
    for marker in range(len(channels)):
        img_data[:, :, marker] = ndimage.gaussian_filter(img_data[:, :, marker],
                                                         sigma=blur_factor)

    # flatten each image
    pixel_mat = img_data.reshape(-1, len(channels))

    # convert into a dataframe
    pixel_mat = pd.DataFrame(pixel_mat, columns=channels)

    # assign metadata about each entry
    pixel_mat['fov'] = fov
    pixel_mat['row_index'] = np.repeat(range(img_data.shape[0]), img_data.shape[1])
    pixel_mat['column_index'] = np.tile(range(img_data.shape[0]), img_data.shape[1])

    # assign segmentation label
    seg_labels_flat = seg_labels.flatten()
    pixel_mat['segmentation_label'] = seg_labels_flat

    # remove any rows that sum to 0
    pixel_mat = pixel_mat.loc[pixel_mat.loc[:, channels].sum(axis=1) != 0, :]

    # normalize each row by total marker counts to convert into frequencies
    pixel_mat.loc[:, channels] = pixel_mat.loc[:, channels].div(
        pixel_mat.loc[:, channels].sum(axis=1), axis=0)

    # subset the pixel matrix for training
    pixel_mat_subset = pixel_mat.sample(frac=subset_proportion, random_state=seed)

    return pixel_mat, pixel_mat_subset


def create_pixel_matrix(fovs, channels, base_dir, tiff_dir, seg_dir,
                        pre_dir='pixel_mat_preprocessed',
                        sub_dir='pixel_mat_subsetted', is_mibitiff=False,
                        blur_factor=2, subset_proportion=0.1, seed=42):
    """Preprocess the images for FlowSOM clustering and creates a pixel-level matrix

    Saves preprocessed data to pre_dir and subsetted data to sub_dir

    Args:
        fovs (list):
            List of fovs to subset over
        channels (list):
            List of channels to subset over
        base_dir (str):
            Name of the directory to save the pixel files to
        tiff_dir (str):
            Name of the directory containing the tiff files
        seg_dir (str):
            Name of the directory containing the segmented files
        pre_dir (str):
            Name of the directory which contains the preprocessed pixel data
        sub_dir (str):
            The name of the directory containing the subsetted pixel data
        is_mibitiff (bool):
            Whether to load the images from MIBITiff
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

    if not os.path.exists(tiff_dir):
        raise FileNotFoundError("tiff_dir %s does not exist" % tiff_dir)

    # create pre_dir if it doesn't already exist
    if not os.path.exists(os.path.join(base_dir, pre_dir)):
        os.mkdir(os.path.join(base_dir, pre_dir))

    # create sub_dir if it doesn't already exist
    if not os.path.exists(os.path.join(base_dir, sub_dir)):
        os.mkdir(os.path.join(base_dir, sub_dir))

    # iterate over fov_batches
    for fov in fovs:
        # load img_xr from MIBITiff or directory with the fov
        if is_mibitiff:
            img_xr = load_utils.load_imgs_from_mibitiff(
                tiff_dir, mibitiff_files=[fov], channels=channels, dtype="int16"
            )
        else:
            img_xr = load_utils.load_imgs_from_tree(
                tiff_dir, fovs=[fov], channels=channels, dtype="int16"
            )

        # load segmentation labels in for fov
        seg_labels = imread(os.path.join(seg_dir, fov + '_feature_0.tif'))

        # subset for the channel data
        img_data = img_xr.loc[fov, ...].values.astype(np.float32)

        # create the full and subsetted fov matrices
        pixel_mat, pixel_mat_subset = create_fov_pixel_data(
            fov=fov, channels=channels, img_data=img_data, seg_labels=seg_labels,
            blur_factor=blur_factor, subset_proportion=subset_proportion, seed=seed
        )

        # write complete dataset to feather, needed for cluster assignment
        feather.write_dataframe(pixel_mat,
                                os.path.join(base_dir,
                                             pre_dir,
                                             fov + ".feather"),
                                compression='uncompressed')

        # write subseted dataset to feather, needed for training
        feather.write_dataframe(pixel_mat_subset,
                                os.path.join(base_dir,
                                             sub_dir,
                                             fov + ".feather"),
                                compression='uncompressed')


def train_som(fovs, channels, base_dir,
              sub_dir='pixel_mat_subsetted', norm_vals_name='norm_vals.feather',
              weights_name='weights.feather', num_passes=1):
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
        norm_vals_name (str):
            The name of the file to store the 99.9% normalized values
        weights_name (str):
            The name of the weights file
        num_passes (int):
            The number of training passes to make through the dataset
    """

    # define the paths to the data
    subsetted_path = os.path.join(base_dir, sub_dir)
    norm_vals_path = os.path.join(base_dir, norm_vals_name)
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
                    str(num_passes), subsetted_path, norm_vals_path, weights_path]
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
                   norm_vals_name='norm_vals.feather', weights_name='weights.feather',
                   cluster_dir='pixel_mat_clustered'):
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
        norm_vals_name (str):
            The name of the file to store the 99.9% normalized values
        weights_name (str):
            The name of the weights file
        cluster_dir (str):
            The name of the directory to write the clustered data
    """

    # define the paths to the data
    preprocessed_path = os.path.join(base_dir, pre_dir)
    norm_vals_path = os.path.join(base_dir, norm_vals_name)
    weights_path = os.path.join(base_dir, weights_name)
    clustered_path = os.path.join(base_dir, cluster_dir)

    # if path to the preprocessed file does not exist
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError('Pixel preprocessed directory %s does not exist in base_dir %s' %
                                (pre_dir, base_dir))

    if not os.path.exists(norm_vals_path):
        raise FileNotFoundError('Normalized values file %s does not exist in base_dir %s' %
                                (norm_vals_path, base_dir))

    # if path to the weights file does not exist
    if not os.path.exists(weights_path):
        raise FileNotFoundError('Weights file %s does not exist in base_dir %s' %
                                (weights_name, base_dir))

    # verify that all provided fovs exist in the folder
    files = io_utils.list_files(preprocessed_path, substrs='.feather')
    misc_utils.verify_in_list(provided_fovs=fovs,
                              subsetted_fovs=io_utils.remove_file_extensions(files))

    # ensure the norm vals columns are valid indexes
    norm_vals = feather.read_dataframe(os.path.join(base_dir, norm_vals_name))
    sample_fov = feather.read_dataframe(os.path.join(base_dir, pre_dir, files[0]))
    misc_utils.verify_in_list(norm_vals_columns=norm_vals.columns.values,
                              pixel_data_columns=sample_fov.columns.values)

    # ensure the weights columns are valid indexes
    weights = feather.read_dataframe(os.path.join(base_dir, weights_name))
    misc_utils.verify_in_list(weights_columns=weights.columns.values,
                              pixel_data_columns=sample_fov.columns.values)

    # make the clustered dir if it does not exist
    if not os.path.exists(clustered_path):
        os.mkdir(clustered_path)

    # run the trained SOM on the dataset, assigning clusters
    process_args = ['Rscript', '/run_trained_som.R', ','.join(fovs),
                    preprocessed_path, norm_vals_path, weights_path, clustered_path]

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
                      consensus_dir='pixel_mat_consensus'):
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
        consensus_dir (str):
            Name of directory to save the consensus clustered results
    """

    clustered_path = os.path.join(base_dir, cluster_dir)
    cluster_avg_path = os.path.join(base_dir, cluster_avg_name)
    consensus_path = os.path.join(base_dir, consensus_dir)

    if not os.path.exists(clustered_path):
        raise FileNotFoundError('Cluster dir %s does not exist in base_dir %s' %
                                (base_dir, clustered_path))

    # compute and write the averaged cluster results
    compute_cluster_avg(fovs, channels, base_dir, cluster_dir)

    # make consensus_dir if it doesn't exist
    if not os.path.exists(consensus_path):
        os.mkdir(consensus_path)

    # run the consensus clustering process
    process_args = ['Rscript', '/consensus_cluster.R', ','.join(fovs), ','.join(channels),
                    str(max_k), str(cap), clustered_path, cluster_avg_path, consensus_path]

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
