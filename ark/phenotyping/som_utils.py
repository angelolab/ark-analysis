import os
import subprocess

import feather
import numpy as np
import pandas as pd
import re
import scipy.ndimage as ndimage
from skimage.io import imread
import xarray as xr

import ark.settings as settings
from ark.analysis import visualize
from ark.utils import io_utils
from ark.utils import load_utils
from ark.utils import misc_utils


def compute_pixel_cluster_avg(fovs, channels, base_dir, cluster_col,
                              cluster_dir='pixel_mat_clustered'):
    """For each fov, compute the average channel values across each pixel SOM cluster

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directories
        cluster_col (str):
            Name of the column to group by
        cluster_dir (str):
            Name of the file containing the pixel data with cluster labels

    Returns:
        pandas.DataFrame:
            Contains the average channel values for each pixel SOM cluster
    """

    # define the cluster averages DataFrame
    cluster_avgs = pd.DataFrame()

    for fov in fovs:
        # read in the fovs data
        fov_pixel_data = feather.read_dataframe(
            os.path.join(base_dir, cluster_dir, fov + '.feather')
        )

        # aggregate the sums and counts
        sum_by_cluster = fov_pixel_data.groupby(cluster_col)[channels].sum()
        count_by_cluster = fov_pixel_data.groupby(cluster_col)[channels].size().to_frame('count')

        # concat the results together
        agg_results = pd.merge(
            sum_by_cluster, count_by_cluster, left_index=True, right_index=True).reset_index()

        cluster_avgs = pd.concat([cluster_avgs, agg_results])

    # sum the counts and the channel sums
    sum_count_totals = cluster_avgs.groupby(cluster_col)[channels + ['count']].sum().reset_index()

    # now compute the means using the count column
    sum_count_totals[channels] = sum_count_totals[channels].div(sum_count_totals['count'], axis=0)

    # drop the count column
    sum_count_totals = sum_count_totals.drop('count', axis=1)

    return sum_count_totals


def compute_cell_cluster_avg(cluster_path, column_prefix, cluster_col):
    """For each cell SOM cluster, compute the average number of associated pixel SOM/meta clusters

    Args:
        cluster_path (str):
            The path to the cell data with SOM labels
        column_prefix (str):
            The prefix of the columns to subset, should be 'cluster' or 'hCluster_cap'
        cluster_col (str):
            Name of the cell cluster column to group by

    Returns:
        pandas.DataFrame:
            Contains the average values for each column across cell SOM clusters
    """

    # read in the clustered data
    cluster_data = feather.read_dataframe(cluster_path)

    # subset by columns with cluster in them
    column_subset = [c for c in cluster_data.columns.values if c.startswith(column_prefix + '_')]
    cluster_data_subset = cluster_data.loc[:, column_subset + [cluster_col]]

    # average each column
    mean_count_totals = cluster_data_subset.groupby(cluster_col).mean().reset_index()

    return mean_count_totals


def compute_cell_cluster_counts(fovs, channels, base_dir, consensus_dir,
                                cell_table_path, cluster_col='cluster'):
    """Create a matrix with each fov-cell label pair and their pixel SOM/meta cluster counts

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directories
        consensus_dir (str):
            Name of directory to save the consensus clustered results
        cell_table_path (str):
            Path to the cell table, needs to be created with Segment_Image_Data.ipynb
        cluster_col (str):
            The name of the cluster column to group by, should be 'cluster' or 'hCluster_cap'

    Returns:
        pd.DataFrame:
            cell x cluster list of counts of each pixel SOM/meta cluster per each cell
    """

    # define the counts per segmentation label DataFrame
    cluster_counts = pd.DataFrame()

    # read the cell table data
    cell_table = pd.read_csv(cell_table_path)

    # subset on fov, label, and cell size
    cell_table = cell_table[['fov', 'label', 'cell_size']]

    # for verification purposes, make sure the fovs are sorted in numerical order
    fovs_sorted = sorted(fovs, key=lambda x: int(re.findall(r'\d+', x)[0]))

    for fov in fovs_sorted:
        # read the full dataset with pixel and meta clusters
        fov_pixel_data = feather.read_dataframe(
            os.path.join(base_dir, consensus_dir, fov + '.feather')
        )

        # discard fov as we'll get it from the cell table
        fov_pixel_data = fov_pixel_data.drop(columns='fov')

        # group the data by segmentation label and cluster value
        group_by_cluster_col = fov_pixel_data.groupby(
            ['segmentation_label', cluster_col]
        ).size().reset_index(name='count')

        # cast the cluster column to int for aesthetics
        group_by_cluster_col[cluster_col] = group_by_cluster_col[cluster_col].astype(int)

        # create a pivot table with counts of each cluster per segmentation label
        num_cluster_per_seg_label = group_by_cluster_col.pivot(
            index='segmentation_label', columns=cluster_col, values='count'
        ).fillna(0).astype(int)

        # change column names to indicate which cluster type (pixel or meta) is used
        new_columns = ['%s_' % cluster_col + str(c) for c in num_cluster_per_seg_label.columns]
        num_cluster_per_seg_label.columns = new_columns

        # copy over the segmentation label data
        num_cluster_per_seg_label['segmentation_label'] = num_cluster_per_seg_label.index.values

        # reset the index
        num_cluster_per_seg_label = num_cluster_per_seg_label.reset_index(drop=True)

        # subset cell table on specific fov, convert labels to int to match data types
        cell_table_fov = cell_table[cell_table['fov'] == fov]
        cell_table_fov['label'] = cell_table_fov['label'].astype(int)

        # rename label in cell_table_fov as segmentation_label for joining purposes
        cell_table_fov = cell_table_fov.rename(columns={'label': 'segmentation_label'})

        # join the two DataFrames
        num_cluster_per_seg_label = num_cluster_per_seg_label.merge(
            cell_table_fov,
            how='inner',
            on=['segmentation_label']
        )

        # concatenate to cluster_counts
        cluster_counts = pd.concat([cluster_counts, num_cluster_per_seg_label])

    # fill empty elements with 0
    cluster_counts = cluster_counts.fillna(0)

    # remove any clusters that sum to 0
    cluster_columns = [c for c in cluster_counts.columns.values if c.startswith(cluster_col)]
    zero_columns = cluster_counts.loc[:, cluster_columns].columns[
        cluster_counts.loc[:, cluster_columns].sum() == 0
    ]
    cluster_counts = cluster_counts.drop(columns=zero_columns)

    # because cell_table is quite large, free up memory space
    del cell_table

    return cluster_counts


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
            Contains the following:

            - pandas.DataFrame: Gaussian blurred and channel sum normalized pixel data for a fov
            - pandas.DataFrame: subset of the preprocessed pixel dataset for a fov
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

    # subset the pixel matrix for training
    pixel_mat_subset = pixel_mat.sample(frac=subset_proportion, random_state=seed)

    return pixel_mat, pixel_mat_subset


def create_pixel_matrix(fovs, base_dir, tiff_dir, seg_dir,
                        pre_dir='pixel_mat_preprocessed',
                        sub_dir='pixel_mat_subsetted', is_mibitiff=False,
                        blur_factor=2, subset_proportion=0.1, seed=42):
    """For each fov, add a Gaussian blur to each channel and normalize channel sums for each pixel

    Saves data to pre_dir and subsetted data to sub_dir

    Args:
        fovs (list):
            List of fovs to subset over
        base_dir (str):
            The path to the data directories
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
                tiff_dir, mibitiff_files=[fov], dtype="int16"
            )
        else:
            img_xr = load_utils.load_imgs_from_tree(
                tiff_dir, fovs=[fov], dtype="int16"
            )

        # load segmentation labels in for fov
        seg_labels = imread(os.path.join(seg_dir, fov + '_feature_0.tif'))

        # subset for the channel data
        img_data = img_xr.loc[fov, ...].values.astype(np.float32)

        # create the full and subsetted fov matrices
        pixel_mat, pixel_mat_subset = create_fov_pixel_data(
            fov=fov, channels=img_xr.channels.values, img_data=img_data, seg_labels=seg_labels,
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


def train_pixel_som(fovs, channels, base_dir,
                    sub_dir='pixel_mat_subsetted', norm_vals_name='norm_vals.feather',
                    weights_name='pixel_weights.feather', xdim=10, ydim=10,
                    lr_start=0.05, lr_end=0.01, num_passes=1, seed=42):
    """Run the SOM training on the subsetted pixel data.

    Saves weights to base_dir/weights_name.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of markers to subset on
        base_dir (str):
            The path to the data directories
        sub_dir (str):
            The name of the subsetted data directory
        norm_vals_name (str):
            The name of the file to store the 99.9% normalized values
        weights_name (str):
            The name of the weights file
        xdim (int):
            The number of x nodes to use for the SOM
        ydim (int):
            The number of y nodes to use for the SOM
        lr_start (float):
            The start learning rate for the SOM, decays to lr_end
        lr_end (float):
            The end learning rate for the SOM, decays from lr_start
        num_passes (int):
            The number of training passes to make through the dataset
        seed (int):
            The random seed to set for training
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
    process_args = ['Rscript', '/create_pixel_som.R', ','.join(fovs), ','.join(channels),
                    str(xdim), str(ydim), str(lr_start), str(lr_end), str(num_passes),
                    subsetted_path, norm_vals_path, weights_path, str(seed)]

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
    process_args = ['Rscript', '/run_pixel_som.R', ','.join(fovs),
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


def pixel_consensus_cluster(fovs, channels, base_dir, max_k=20, cap=3,
                            cluster_dir='pixel_mat_clustered',
                            cluster_avg_name='pixel_cluster_avg.feather',
                            consensus_dir='pixel_mat_consensus', seed=42):
    """Run consensus clustering algorithm on pixel-level summed data across channels

    Saves data with consensus cluster labels to consensus_dir

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directory
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
        seed (int):
            The random seed to set for consensus clustering
    """

    clustered_path = os.path.join(base_dir, cluster_dir)
    cluster_avg_path = os.path.join(base_dir, cluster_avg_name)
    consensus_path = os.path.join(base_dir, consensus_dir)

    if not os.path.exists(clustered_path):
        raise FileNotFoundError('Cluster dir %s does not exist in base_dir %s' %
                                (cluster_dir, base_dir))

    # compute the cluster averages
    cluster_avgs = compute_pixel_cluster_avg(fovs, channels, base_dir,
                                             cluster_col='cluster', cluster_dir=cluster_dir)

    # save the cluster averages
    feather.write_dataframe(cluster_avgs,
                            os.path.join(base_dir, cluster_avg_name),
                            compression='uncompressed')

    # make consensus_dir if it doesn't exist
    if not os.path.exists(consensus_path):
        os.mkdir(consensus_path)

    # run the consensus clustering process
    process_args = ['Rscript', '/pixel_consensus_cluster.R', ','.join(fovs), ','.join(channels),
                    str(max_k), str(cap), clustered_path, cluster_avg_path, consensus_path,
                    str(seed)]

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


def visualize_pixel_cluster_data(fovs, channels, base_dir, cluster_dir,
                                 pixel_cluster_col='cluster', dpi=None, center_val=None,
                                 overlay_values=False, colormap="vlag",
                                 save_dir=None, save_file=None):
    """For pixel-level analysis, visualize the average cluster results for each cluster

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directories
        cluster_dir (str):
            Name of the directory containing the data to visualize
        pixel_cluster_col (str):
            Name of the column to group values by
        dpi (float):
            The resolution of the image to save, ignored if save_dir is None
        center_val (float):
            value at which to center the heatmap
        overlay_values (bool):
            whether to overlay the raw heatmap values on top
        colormap (str):
            color scheme for visualization
        save_dir (str):
            If specified, a directory where we will save the plot
        save_file (str):
            If save_dir specified, specify a file name you wish to save to.
            Ignored if save_dir is None
    """

    # average the channel values across the cluster column
    cluster_avgs = compute_pixel_cluster_avg(fovs, channels, base_dir,
                                             pixel_cluster_col, cluster_dir)

    # convert cluster column to integer type
    cluster_avgs[pixel_cluster_col] = cluster_avgs[pixel_cluster_col].astype(int)

    # sort cluster col in ascending order
    cluster_avgs = cluster_avgs.sort_values(by=pixel_cluster_col)

    # draw the heatmap
    visualize.draw_heatmap(
        data=cluster_avgs[channels].values, x_labels=cluster_avgs[pixel_cluster_col],
        y_labels=channels, dpi=dpi, center_val=center_val, overlay_values=overlay_values,
        colormap=colormap, save_dir=save_dir, save_file=save_file
    )


def train_cell_som(fovs, channels, base_dir, pixel_consensus_dir, cell_table_name,
                   cluster_counts_name='cluster_counts.feather', cluster_col='cluster',
                   weights_name='cell_weights.feather', xdim=10, ydim=10,
                   lr_start=0.05, lr_end=0.01, num_passes=1, seed=42):
    """Run the SOM training on the number of pixel/meta clusters in each cell of each fov

    Saves the weights to base_dir/weights_name

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directories
        pixel_consensus_dir (str):
            Name of directory which contains the pixel-level consensus data
        cell_table_name (str):
            Name of the cell table, needs to be created with Segment_Image_Data.ipynb
        cluster_counts_name (str):
            Name of the file to save the cluster counts of each cell
        cluster_col (str):
            Name of the column with the pixel SOM cluster assignments
        weights_name (str):
            The name of the weights file
        xdim (int):
            The number of x nodes to use for the SOM
        ydim (int):
            The number of y nodes to use for the SOM
        lr_start (float):
            The start learning rate for the SOM, decays to lr_end
        lr_end (float):
            The end learning rate for the SOM, decays from lr_start
        num_passes (int):
            The number of training passes to make through the dataset
        seed (int):
            The random seed to set for training
    """

    # define the data paths
    cell_table_path = os.path.join(base_dir, cell_table_name)
    cluster_counts_path = os.path.join(base_dir, cluster_counts_name)
    weights_path = os.path.join(base_dir, weights_name)

    if not os.path.exists(cell_table_path):
        raise FileNotFoundError('Cell table %s does not exist in base_dir %s' %
                                (cell_table_name, base_dir))

    # generate a matrix with each fov/cell label pair with their pixel SOM/meta cluster counts
    cluster_counts = compute_cell_cluster_counts(
        fovs, channels, base_dir, pixel_consensus_dir, cell_table_path, cluster_col
    )

    # write the created matrix
    feather.write_dataframe(cluster_counts,
                            os.path.join(base_dir, cluster_counts_name),
                            compression='uncompressed')

    # run the SOM training process
    process_args = ['Rscript', '/create_cell_som.R', ','.join(fovs), str(xdim), str(ydim),
                    str(lr_start), str(lr_end), str(num_passes), cluster_counts_path,
                    weights_path, str(seed)]

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


def cluster_cells(base_dir, cluster_counts_name='cluster_counts.feather',
                  weights_name='cell_weights.feather', cell_cluster_name='cell_clusters.feather'):
    """Uses trained weights to assign cluster labels on full pixel data

    Saves data with cluster labels to cell_cluster_name

    Args:
        base_dir (str):
            The path to the data directory
        cluster_counts_name (str):
            Name of the file with the cluster counts of each cell
        weights_name (str):
            The name of the weights file
        cell_cluster_name (str):
            The name of the file to write the clustered data
    """

    # define the paths to the data
    cluster_counts_path = os.path.join(base_dir, cluster_counts_name)
    weights_path = os.path.join(base_dir, weights_name)
    cell_cluster_path = os.path.join(base_dir, cell_cluster_name)

    if not os.path.exists(cluster_counts_path):
        raise FileNotFoundError('Cluster counts table %s does not exist in base_dir %s' %
                                (cluster_counts_name, base_dir))

    if not os.path.exists(weights_path):
        raise FileNotFoundError('Weights file %s does not exist in base_dir %s' %
                                (weights_name, base_dir))

    # run the trained SOM on the dataset, assigning clusters
    process_args = ['Rscript', '/run_cell_som.R', cluster_counts_path,
                    weights_path, cell_cluster_path]

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


def cell_consensus_cluster(base_dir, max_k=20, cap=3, column_prefix='cluster',
                           cell_cluster_name='cell_mat_clustered.feather',
                           cell_cluster_avg_name='cell_cluster_avg.feather',
                           cell_consensus_name='cell_mat_consensus.feather', seed=42):
    """Run consensus clustering algorithm on cell-level summed data across channels

    Saves data with consensus cluster labels to cell_consensus_name

    Args:
        base_dir (str):
            The path to the data directory
        max_k (int):
            The number of consensus clusters
        cap (int):
            z-score cap to use when hierarchical clustering
        column_prefix (str):
            The prefix of the columns to subset, should be 'cluster' or 'hCluster_cap'
        cell_cluster_name (str):
            Name of the file containing the cell data with cluster labels
        cell_cluster_avg_name (str):
            Name of file to save the column-averaged results to
        cell_consensus_name (str):
            Name of file to save the consensus clustered results
        seed (int):
            The random seed to set for consensus clustering
    """

    clustered_path = os.path.join(base_dir, cell_cluster_name)
    cluster_avg_path = os.path.join(base_dir, cell_cluster_avg_name)
    consensus_path = os.path.join(base_dir, cell_consensus_name)

    if not os.path.exists(clustered_path):
        raise FileNotFoundError('Cluster table %s does not exist in base_dir %s' %
                                (cluster_dir, base_dir))

    # compute the cluster averages
    cluster_avgs = compute_cell_cluster_avg(clustered_path, column_prefix=column_prefix,
                                            cluster_col='cluster')

    # save the cluster averages
    feather.write_dataframe(cluster_avgs,
                            os.path.join(base_dir, cell_cluster_avg_name),
                            compression='uncompressed')

    # run the consensus clustering process
    process_args = ['Rscript', '/cell_consensus_cluster.R', str(max_k), str(cap), clustered_path,
                    cluster_avg_path, consensus_path, str(seed)]

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


def visualize_cell_cluster_data(base_dir, cluster_name, column_prefix, cell_cluster_col='cluster',
                                dpi=None, center_val=None, overlay_values=False, colormap="vlag",
                                save_dir=None, save_file=None):
    """For cell-level analysis, visualize the average cluster results for each cluster

    Args:
        base_dir (str):
            The path to the data directories
        cluster_name (str):
            The name of the file containing the cluster data
        column_prefix (str):
            The prefix of the columns to subset, should be 'cluster' or 'hCluster_cap'
        cell_cluster_col (str):
            Name of the column to group values by
        dpi (float):
            The resolution of the image to save, ignored if save_dir is None
        center_val (float):
            value at which to center the heatmap
        overlay_values (bool):
            whether to overlay the raw heatmap values on top
        colormap (str):
            color scheme for visualization
        save_dir (str):
            If specified, a directory where we will save the plot
        save_file (str):
            If save_dir specified, specify a file name you wish to save to.
            Ignored if save_dir is None
    """

    # average the columns across the cluster column
    cluster_avgs = compute_cell_cluster_avg(os.path.join(base_dir, cluster_name),
                                            column_prefix, cell_cluster_col)

    # convert cluster column to integer type
    cluster_avgs[cell_cluster_col] = cluster_avgs[cell_cluster_col].astype(int)

    # sort cluster col in ascending order
    cluster_avgs = cluster_avgs.sort_values(by=cell_cluster_col)

    # draw the heatmap
    visualize.draw_heatmap(
        data=cluster_avgs.drop(columns=cell_cluster_col).values,
        x_labels=cluster_avgs[cell_cluster_col],
        y_labels=cluster_avgs.drop(columns=cell_cluster_col).columns.values,
        dpi=dpi, center_val=center_val, overlay_values=overlay_values,
        colormap=colormap, save_dir=save_dir, save_file=save_file
    )
