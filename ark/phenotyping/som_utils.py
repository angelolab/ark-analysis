import os
import subprocess

import feather
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import scipy.ndimage as ndimage
import scipy.stats as stats
from skimage.io import imread
import xarray as xr

from ark.analysis import visualize
import ark.settings as settings
from ark.utils import io_utils
from ark.utils import load_utils
from ark.utils import misc_utils


def normalize_rows(pixel_data, channels, include_seg_label=True):
    """Normalizes the rows of a pixel matrix by their sum

    Args:
        pixel_data (pandas.DataFrame):
            The dataframe containing the pixel data for a given fov
            Includes channel and meta (`fov`, `segmentation_label`, etc.) columns
        channels (list):
            List of channels to subset over
        include_seg_label (bool):
            Whether to include `'segmentation_label'` as a metadata column

    Returns:
        pandas.DataFrame:
            The pixel data with rows normalized and 0-sum rows removed
    """

    # subset the fov data by the channels the user trained the pixel SOM on
    pixel_data_sub = pixel_data[channels]

    # divide each row by their sum
    pixel_data_sub = pixel_data_sub.div(pixel_data_sub.sum(axis=1), axis=0)

    # define the meta columns to add back
    meta_cols = ['fov', 'row_index', 'column_index']

    # add the segmentation_label column if it should be kept
    if include_seg_label:
        meta_cols.append('segmentation_label')

    # add back meta columns, making sure to remove 0-row indices
    pixel_data_sub[meta_cols] = pixel_data.loc[pixel_data_sub.index.values, meta_cols]

    return pixel_data_sub


def compute_pixel_cluster_channel_avg(fovs, channels, base_dir, pixel_cluster_col,
                                      pixel_cluster_dir='pixel_mat_clustered', keep_count=False):
    """Compute the average channel values across each pixel SOM cluster

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directories
        pixel_cluster_col (str):
            Name of the column to group by
        pixel_cluster_dir (str):
            Name of the directory containing the pixel data with cluster labels
        keep_count (bool):
            Whether to keep the count column when aggregating or not
            This should only be set to `True` for visualization purposes

    Returns:
        pandas.DataFrame:
            Contains the average channel values for each pixel SOM/meta cluster
    """

    # verify the pixel cluster col specified is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster']
    )

    # define the cluster averages DataFrame
    cluster_avgs = pd.DataFrame()

    for fov in fovs:
        # read in the fovs data
        fov_pixel_data = feather.read_dataframe(
            os.path.join(base_dir, pixel_cluster_dir, fov + '.feather')
        )

        # aggregate the sums and counts
        sum_by_cluster = fov_pixel_data.groupby(
            pixel_cluster_col
        )[channels].sum()
        count_by_cluster = fov_pixel_data.groupby(
            pixel_cluster_col
        )[channels].size().to_frame('count')

        # merge the results by column
        agg_results = pd.merge(
            sum_by_cluster, count_by_cluster, left_index=True, right_index=True
        ).reset_index()

        # concat the results together
        cluster_avgs = pd.concat([cluster_avgs, agg_results])

    # reset the index of cluster_avgs for consistency
    cluster_avgs = cluster_avgs.reset_index(drop=True)

    # sum the counts and the channel sums
    sum_count_totals = cluster_avgs.groupby(
        pixel_cluster_col
    )[channels + ['count']].sum().reset_index()

    # now compute the means using the count column
    sum_count_totals[channels] = sum_count_totals[channels].div(sum_count_totals['count'], axis=0)

    # convert cluster column to integer type
    sum_count_totals[pixel_cluster_col] = sum_count_totals[pixel_cluster_col].astype(int)

    # sort cluster col in ascending order
    sum_count_totals = sum_count_totals.sort_values(by=pixel_cluster_col)

    # drop the count column if specified
    if not keep_count:
        sum_count_totals = sum_count_totals.drop('count', axis=1)

    return sum_count_totals


def compute_cell_cluster_count_avg(cell_cluster_path, pixel_cluster_col_prefix,
                                   cell_cluster_col, keep_count=False):
    """For each cell SOM cluster, compute the average number of associated pixel SOM/meta clusters

    Args:
        cell_cluster_path (str):
            The path to the cell data with SOM and/or meta labels, created by `cluster_cells`
        pixel_cluster_col_prefix (str):
            The prefix of the pixel cluster count columns to subset,
            should be `'pixel_som_cluster'` or `'pixel_meta_cluster'`
        cell_cluster_col (str):
            Name of the cell cluster column to group by
            should be `'cell_som_cluster'` or `'cell_meta_cluster'`
        keep_count (bool):
            Whether to include the cell counts or not,
            this should only be set to `True` for visualization purposes

    Returns:
        pandas.DataFrame:
            Contains the average values for each column across cell SOM clusters
    """

    # verify the pixel cluster col prefix specified is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col_prefix],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster_rename']
    )

    # verify the cell cluster col prefix specified is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[cell_cluster_col],
        valid_cluster_cols=['cell_som_cluster', 'cell_meta_cluster']
    )

    # read in the clustered data
    cluster_data = feather.read_dataframe(cell_cluster_path)

    # subset by columns with cluster in them
    column_subset = [
        c for c in cluster_data.columns.values if c.startswith(pixel_cluster_col_prefix + '_')
    ]
    cluster_data_subset = cluster_data.loc[:, column_subset + [cell_cluster_col]]

    # average each column grouped by the cell cluster column
    mean_count_totals = cluster_data_subset.groupby(cell_cluster_col).mean().reset_index()

    # if keep_count is included, add the count column to the cell table
    if keep_count:
        cell_cluster_totals = cluster_data_subset.groupby(
            cell_cluster_col
        ).size().to_frame('count')
        cell_cluster_totals = cell_cluster_totals.reset_index(drop=True)
        mean_count_totals['count'] = cell_cluster_totals['count']

    return mean_count_totals


def compute_cell_cluster_channel_avg(fovs, channels, base_dir,
                                     weighted_cell_channel_name,
                                     cell_cluster_name='cell_mat_clustered.feather',
                                     cell_cluster_col='cell_meta_cluster'):
    """Computes the average marker expression for each cell cluster

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directory
        weighted_cell_channel_name (str):
            The name of the weighted cell table, created in `example_cell_clustering.ipynb`
        cell_cluster_name (str):
            Name of the file containing the cell data with cluster labels
        cell_cluster_col (str):
            Whether to aggregate by cell SOM or meta labels
            Needs to be either 'cell_som_cluster', or 'cell_meta_cluster'

    Returns:
        pandas.DataFrame:
            Each cell cluster mapped to the average expression for each marker
    """

    # verify the cell table actually exists
    if not os.path.exists(os.path.join(base_dir, weighted_cell_channel_name)):
        raise FileNotFoundError(
            "Weighted cell table %s not found in %s" % (weighted_cell_channel_name, base_dir)
        )

    # verify the cell cluster col specified is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[cell_cluster_col],
        valid_cluster_cols=['cell_som_cluster', 'cell_meta_cluster']
    )

    # read the weighted cell channel table in
    cell_table = pd.read_csv(os.path.join(base_dir, weighted_cell_channel_name))

    # subset on only the fovs the user has specified
    cell_table = cell_table[cell_table['fov'].isin(fovs)]

    # read the clustered data
    cluster_data = feather.read_dataframe(os.path.join(base_dir, cell_cluster_name))

    # need to ensure that both cell_table and cluster_data have FOVs and segmentation_labels sorted
    # in the same order, this can be done by simply sorting by fov and segmentation_label for both
    cell_table = cell_table.sort_values(
        by=['fov', 'segmentation_label']
    ).reset_index(drop=True)
    cluster_data = cluster_data.sort_values(
        by=['fov', 'segmentation_label']
    ).reset_index(drop=True)

    # add an extra check to ensure that the FOVs and segmentation labels are in the same order
    misc_utils.verify_same_elements(
        enforce_order=True,
        cell_table_fovs=list(cell_table['fov']),
        cluster_data_fovs=list(cluster_data['fov'])
    )
    misc_utils.verify_same_elements(
        enforce_order=True,
        cell_table_labels=list(cell_table['segmentation_label']),
        cluster_data_labels=list(cluster_data['segmentation_label'])
    )

    # assign the cluster labels to cell_table
    cell_table[cell_cluster_col] = cluster_data[cell_cluster_col]

    # subset the cell table by just the desired channels and the cell_cluster_col
    cell_table = cell_table[channels + [cell_cluster_col]]

    # compute the mean channel expression across each cell cluster
    channel_avgs = cell_table.groupby(cell_cluster_col).mean().reset_index()

    return channel_avgs


def compute_p2c_weighted_channel_avg(pixel_channel_avg, channels, cell_counts,
                                     fovs=None, pixel_cluster_col='pixel_meta_cluster_rename'):
    """Compute the average marker expression for each cell weighted by pixel cluster

    This expression is weighted by the pixel SOM/meta cluster counts. So for each cell,
    marker expression vector is computed by:

    `pixel_cluster_n_count * avg_marker_exp_pixel_cluster_n + ...`

    These values are then normalized by the cell's respective size.

    Note that this function will only be used to correct overlapping signal for visualization.

    Args:
        pixel_channel_avg (pandas.DataFrame):
            The average channel values for each pixel SOM/meta cluster
            Computed by `compute_pixel_cluster_channel_avg`
        channels (list):
            The list of channels to subset `pixel_channel_avg` by
        cell_counts (pandas.DataFrame):
            The dataframe listing the number of each type of pixel SOM/meta cluster per cell
        fovs (list):
            The list of fovs to include, if `None` provided all are used
        pixel_cluster_col (str):
            Name of the cell cluster column to group by
            Should be `'pixel_som_cluster'` or `'pixel_meta_cluster_rename'`

    Returns:
        pandas.DataFrame:
            Returns the average marker expression for each cell in the dataset
    """

    # if no fovs provided make sure they're all iterated over
    if fovs is None:
        fovs = list(cell_counts['fov'].unique())

    # verify that the fovs provided are valid
    misc_utils.verify_in_list(
        provided_fovs=fovs,
        dataset_fovs=cell_counts['fov'].unique()
    )

    # verify the pixel_cluster_col provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster_rename']
    )

    # subset over the provided fovs
    cell_counts_sub = cell_counts[cell_counts['fov'].isin(fovs)].copy()

    # subset over the cluster count columns of pixel_channel_avg
    cluster_cols = [c for c in cell_counts_sub.columns.values if pixel_cluster_col in c]
    cell_counts_clusters = cell_counts_sub[cluster_cols].copy()

    # sort the columns of cell_counts_clusters in ascending cluster order
    cell_counts_clusters = cell_counts_clusters.reindex(
        sorted(cell_counts_clusters.columns.values),
        axis=1
    )

    # sort the pixel_channel_avg table by pixel_cluster_col in ascending cluster order
    # NOTE: to handle numeric cluster names types, we need to cast the pixel_cluster_col values
    # to str to ensure the same sorting is used
    if pixel_channel_avg[pixel_cluster_col].dtype == int:
        pixel_channel_avg[pixel_cluster_col] = pixel_channel_avg[pixel_cluster_col].astype(str)

    pixel_channel_avg_sorted = pixel_channel_avg.sort_values(by=pixel_cluster_col)

    # check that the same clusters are in both cell_counts_clusters and pixel_channel_avg_sorted
    # the matrix multiplication will fail if this is not caught
    cell_counts_cluster_ids = [
        x.replace(pixel_cluster_col + '_', '') for x in cell_counts_clusters.columns.values
    ]

    pixel_channel_cluster_ids = pixel_channel_avg_sorted[pixel_cluster_col].values

    misc_utils.verify_same_elements(
        enforce_order=True,
        cell_counts_cluster_ids=cell_counts_cluster_ids,
        pixel_channel_cluster_ids=pixel_channel_cluster_ids
    )

    # assert that the channel subset provided is valid
    # this should never fail, just as an added protection
    misc_utils.verify_in_list(
        provided_channels=channels,
        pixel_channel_avg_cols=pixel_channel_avg_sorted.columns.values
    )

    # subset over just the markers of pixel_channel_avg
    pixel_channel_avg_sub = pixel_channel_avg_sorted[channels]

    # broadcast multiply cell_counts_clusters and pixel_channel_avg to get weighted
    # average expression values for each cell
    weighted_cell_channel = np.matmul(
        cell_counts_clusters.values, pixel_channel_avg_sub.values
    )

    # convert back to dataframe
    weighted_cell_channel = pd.DataFrame(
        weighted_cell_channel, columns=channels
    )

    # add columns back
    meta_cols = ['cell_size', 'fov', 'segmentation_label']
    weighted_cell_channel[meta_cols] = cell_counts_sub.reset_index(drop=True)[meta_cols]

    # normalize the channel columns by the cell size
    weighted_cell_channel[channels] = weighted_cell_channel[channels].div(
        weighted_cell_channel['cell_size'],
        axis=0
    )

    return weighted_cell_channel


def create_c2pc_data(fovs, pixel_consensus_path,
                     cell_table_path, pixel_cluster_col='pixel_meta_cluster_rename'):
    """Create a matrix with each fov-cell label pair and their SOM pixel/meta cluster counts

    Args:
        fovs (list):
            The list of fovs to subset on
        pixel_consensus_path (str):
            Path to directory with the pixel SOM and meta labels
            Created by `pixel_consensus_cluster`
        cell_table_path (str):
            Path to the cell table, needs to be created with `Segment_Image_Data.ipynb`
        pixel_cluster_col (str):
            The name of the pixel cluster column to count per cell
            Should be `'pixel_som_cluster'` or `'pixel_meta_cluster_rename'`

    Returns:
        tuple:

        - `pandas.DataFrame`: cell x cluster counts of each pixel SOM/meta cluster per each cell
        - `pandas.DataFrame`: same as above, but normalized by `cell_size`
    """

    # verify the pixel_cluster_col provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster_rename']
    )

    # read the cell table data
    cell_table = pd.read_csv(cell_table_path)

    # verify that the user has specified fov, label, and cell_size columns in their cell table
    misc_utils.verify_in_list(
        required_cell_table_cols=['fov', 'label', 'cell_size'],
        provided_cell_table_cols=cell_table.columns.values
    )

    # subset on fov, label, and cell size
    cell_table = cell_table[['fov', 'label', 'cell_size']]

    # convert labels to int type
    cell_table['label'] = cell_table['label'].astype(int)

    # rename cell_table label as segmentation_label for joining purposes
    cell_table = cell_table.rename(columns={'label': 'segmentation_label'})

    # subset on only the fovs the user has specified
    cell_table = cell_table[cell_table['fov'].isin(fovs)]

    # define cell_table columns to subset on for merging
    cell_table_cols = ['fov', 'segmentation_label', 'cell_size']

    for fov in fovs:
        # read in the pixel dataset for the fov
        fov_pixel_data = feather.read_dataframe(
            os.path.join(pixel_consensus_path, fov + '.feather')
        )

        # create a groupby object that aggregates the segmentation_label and the pixel_cluster_col
        # intermediate step for creating a pivot table, makes it easier
        group_by_cluster_col = fov_pixel_data.groupby(
            ['segmentation_label', pixel_cluster_col]
        ).size().reset_index(name='count')

        # if cluster labels end up as float (can happen with numeric types), convert to int
        if group_by_cluster_col[pixel_cluster_col].dtype == float:
            group_by_cluster_col[pixel_cluster_col] = group_by_cluster_col[
                pixel_cluster_col
            ].astype(int)

        # counts number of pixel SOM/meta clusters per cell
        num_cluster_per_seg_label = group_by_cluster_col.pivot(
            index='segmentation_label', columns=pixel_cluster_col, values='count'
        ).fillna(0).astype(int)

        # renames the columns to have 'pixel_som_cluster_' or 'pixel_meta_cluster_rename_' prefix
        new_columns = [
            '%s_' % pixel_cluster_col + str(c) for c in num_cluster_per_seg_label.columns
        ]
        num_cluster_per_seg_label.columns = new_columns

        # get intersection of the segmentation labels between cell_table_indices
        # and num_cluster_per_seg_label
        cell_table_labels = list(cell_table[cell_table['fov'] == fov]['segmentation_label'])
        cluster_labels = list(num_cluster_per_seg_label.index.values)
        label_intersection = list(set(cell_table_labels).intersection(cluster_labels))

        # subset on the label intersection
        num_cluster_per_seg_label = num_cluster_per_seg_label.loc[label_intersection]
        cell_table_indices = pd.Index(
            cell_table[
                (cell_table['fov'] == fov) &
                (cell_table['segmentation_label'].isin(label_intersection))
            ].index.values
        )

        # combine the data of num_cluster_per_seg_label into cell_table_indices
        num_cluster_per_seg_label = num_cluster_per_seg_label.set_index(cell_table_indices)
        cell_table = cell_table.combine_first(num_cluster_per_seg_label)

    # NaN means the cluster wasn't found in the specified fov-cell pair
    cell_table = cell_table.fillna(0)

    # also produce a cell table with counts normalized by cell_size
    cell_table_norm = cell_table.copy()

    count_cols = [c for c in cell_table_norm.columns if '%s_' % pixel_cluster_col in c]
    cell_table_norm[count_cols] = cell_table_norm[count_cols].div(cell_table_norm['cell_size'],
                                                                  axis=0)

    # reset the indices of cell_table and cell_table_norm to make things consistent
    cell_table = cell_table.reset_index(drop=True)
    cell_table_norm = cell_table_norm.reset_index(drop=True)

    return cell_table, cell_table_norm


def create_fov_pixel_data(fov, channels, img_data, seg_labels,
                          blur_factor=2, subset_proportion=0.1):
    """Preprocess pixel data for one fov

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

    Returns:
        tuple:
            Contains the following:

            - `pandas.DataFrame`: Gaussian blurred and channel sum normalized pixel data for a fov
            - `pandas.DataFrame`: subset of the preprocessed pixel dataset for a fov
    """

    # for each marker, compute the Gaussian blur
    for marker in range(len(channels)):
        img_data[:, :, marker] = ndimage.gaussian_filter(img_data[:, :, marker],
                                                         sigma=blur_factor)

    # flatten each image, make sure to subset only on channels
    pixel_mat = img_data.reshape(-1, len(channels))

    # convert into a dataframe
    pixel_mat = pd.DataFrame(pixel_mat, columns=channels)

    # assign metadata about each entry
    pixel_mat['fov'] = fov
    pixel_mat['row_index'] = np.repeat(range(img_data.shape[0]), img_data.shape[1])
    pixel_mat['column_index'] = np.tile(range(img_data.shape[1]), img_data.shape[0])

    # assign segmentation labels if it is not None
    if seg_labels is not None:
        seg_labels_flat = seg_labels.flatten()
        pixel_mat['segmentation_label'] = seg_labels_flat

    # remove any rows with channels that sum to zero prior to sampling
    pixel_mat = pixel_mat.loc[(pixel_mat[channels] != 0).any(1), :].reset_index(drop=True)

    # normalize the row sums of pixel mat
    pixel_mat = normalize_rows(pixel_mat, channels, seg_labels is not None)

    # subset the pixel matrix for training
    pixel_mat_subset = pixel_mat.sample(frac=subset_proportion)

    return pixel_mat, pixel_mat_subset


def create_pixel_matrix(fovs, channels, base_dir, tiff_dir, seg_dir,
                        img_sub_folder="TIFs", seg_suffix='_feature_0.tif',
                        pre_dir='pixel_mat_preprocessed',
                        subset_dir='pixel_mat_subsetted',
                        norm_vals_name='norm_vals.feather', is_mibitiff=False,
                        blur_factor=2, subset_proportion=0.1, dtype="int16", seed=42):
    """For each fov, add a Gaussian blur to each channel and normalize channel sums for each pixel

    Saves data to `pre_dir` and subsetted data to `subset_dir`

    Args:
        fovs (list):
            List of fovs to subset over
        channels (list):
            List of channels to subset over, applies only to `pixel_mat_subset`
        base_dir (str):
            The path to the data directories
        tiff_dir (str):
            Name of the directory containing the tiff files
        seg_dir (str):
            Name of the directory containing the segmented files.
            Set to `None` if no segmentation directory is available or desired.
        img_sub_folder (str):
            Name of the subdirectory inside `tiff_dir` containing the tiff files.
            Set to `None` if there isn't any.
        seg_suffix (str):
            The suffix that the segmentation images use.
            Ignored if `seg_dir` is `None`.
        pre_dir (str):
            Name of the directory which contains the preprocessed pixel data
        subset_dir (str):
            The name of the directory containing the subsetted pixel data
        norm_vals_name (str):
            The name of the file to store the 99.9% normalization values
        is_mibitiff (bool):
            Whether to load the images from MIBITiff
        blur_factor (int):
            The sigma to set for the Gaussian blur
        subset_proportion (float):
            The proportion of pixels to take from each fov
        seed (int):
            The random seed to set for subsetting
        dtype (type):
            The type to load the image segmentation labels in
    """

    # if the subset_proportion specified is out of range
    if subset_proportion <= 0 or subset_proportion > 1:
        raise ValueError('Invalid subset percentage entered: must be in (0, 1]')

    # if the base directory doesn't exist
    if not os.path.exists(base_dir):
        raise FileNotFoundError("base_dir %s does not exist" % base_dir)

    # if the tiff dir doesn't exist
    if not os.path.exists(tiff_dir):
        raise FileNotFoundError("tiff_dir %s does not exist" % tiff_dir)

    # create pre_dir if it doesn't already exist
    if not os.path.exists(os.path.join(base_dir, pre_dir)):
        os.mkdir(os.path.join(base_dir, pre_dir))

    # create subset_dir if it doesn't already exist
    if not os.path.exists(os.path.join(base_dir, subset_dir)):
        os.mkdir(os.path.join(base_dir, subset_dir))

    # create variable for storing 99.9% values
    quant_dat = pd.DataFrame()

    # set seed for subsetting
    np.random.seed(seed)

    # iterate over fov_batches
    for fov in fovs:
        # load img_xr from MIBITiff or directory with the fov
        if is_mibitiff:
            img_xr = load_utils.load_imgs_from_mibitiff(
                tiff_dir, mibitiff_files=[fov], dtype=dtype
            )
        else:
            img_xr = load_utils.load_imgs_from_tree(
                tiff_dir, img_sub_folder=img_sub_folder, fovs=[fov], dtype=dtype
            )

        # ensure the provided channels will actually exist in img_xr
        misc_utils.verify_in_list(
            provided_chans=channels,
            pixel_mat_chans=img_xr.channels.values
        )

        # if seg_dir is None, leave seg_labels as None
        seg_labels = None

        # otherwise, load segmentation labels in for fov
        if seg_dir is not None:
            seg_labels = imread(os.path.join(seg_dir, fov + seg_suffix))

        # subset for the channel data
        img_data = img_xr.loc[fov, :, :, channels].values.astype(np.float32)

        # create the full and subsetted fov matrices
        pixel_mat, pixel_mat_subset = create_fov_pixel_data(
            fov=fov, channels=channels, img_data=img_data, seg_labels=seg_labels,
            blur_factor=blur_factor, subset_proportion=subset_proportion
        )

        # get 99.9% of the full fov matrix
        quant_dat[fov] = pixel_mat[channels].replace(0, np.nan).quantile(q=0.999, axis=0)

        # write complete dataset to feather, needed for cluster assignment
        feather.write_dataframe(pixel_mat,
                                os.path.join(base_dir,
                                             pre_dir,
                                             fov + ".feather"),
                                compression='uncompressed')

        # write subseted dataset to feather, needed for training
        feather.write_dataframe(pixel_mat_subset,
                                os.path.join(base_dir,
                                             subset_dir,
                                             fov + ".feather"),
                                compression='uncompressed')

    # get mean 99.9% across all fovs for all markers
    mean_quant = pd.DataFrame(quant_dat.mean(axis=1))

    # save 99.9% normalization values
    feather.write_dataframe(mean_quant.T,
                            os.path.join(base_dir, norm_vals_name),
                            compression='uncompressed')


def train_pixel_som(fovs, channels, base_dir,
                    subset_dir='pixel_mat_subsetted', norm_vals_name='norm_vals.feather',
                    weights_name='pixel_weights.feather', xdim=10, ydim=10,
                    lr_start=0.05, lr_end=0.01, num_passes=1, seed=42):
    """Run the SOM training on the subsetted pixel data.

    Saves weights to `base_dir/weights_name`.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of markers to subset on
        base_dir (str):
            The path to the data directories
        subset_dir (str):
            The name of the subsetted data directory
        norm_vals_name (str):
            The name of the file to store the 99.9% normalization values
        weights_name (str):
            The name of the file to save the weights to
        xdim (int):
            The number of x nodes to use for the SOM
        ydim (int):
            The number of y nodes to use for the SOM
        lr_start (float):
            The start learning rate for the SOM, decays to `lr_end`
        lr_end (float):
            The end learning rate for the SOM, decays from `lr_start`
        num_passes (int):
            The number of training passes to make through the dataset
        seed (int):
            The random seed to set for training
    """

    # define the paths to the data
    subsetted_path = os.path.join(base_dir, subset_dir)
    norm_vals_path = os.path.join(base_dir, norm_vals_name)
    weights_path = os.path.join(base_dir, weights_name)

    # if path to the subsetted file does not exist
    if not os.path.exists(subsetted_path):
        raise FileNotFoundError('Pixel subsetted directory %s does not exist in base_dir %s' %
                                (subset_dir, base_dir))

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


def cluster_pixels(fovs, channels, base_dir, pre_dir='pixel_mat_preprocessed',
                   norm_vals_name='norm_vals.feather', weights_name='pixel_weights.feather',
                   cluster_dir='pixel_mat_clustered',
                   pc_chan_avg_som_cluster_name='pixel_channel_avg_som_cluster.csv'):
    """Uses trained weights to assign cluster labels on full pixel data

    Saves data with cluster labels to `cluster_dir`. Computes and saves the average channel
    expression across pixel SOM clusters.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directory
        pre_dir (str):
            Name of the directory which contains the preprocessed pixel data
        norm_vals_name (str):
            The name of the file with the 99.9% normalized values, created by `train_pixel_som`
        weights_name (str):
            The name of the weights file created by `train_pixel_som`
        cluster_dir (str):
            The name of the directory to write the clustered data
        pc_chan_avg_som_cluster_name (str):
            The name of the file to save the average channel expression across all SOM clusters
    """

    # define the paths to the data
    preprocessed_path = os.path.join(base_dir, pre_dir)
    norm_vals_path = os.path.join(base_dir, norm_vals_name)
    weights_path = os.path.join(base_dir, weights_name)
    clustered_path = os.path.join(base_dir, cluster_dir)

    # if path to the preprocessed directory does not exist
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError('Pixel preprocessed directory %s does not exist in base_dir %s' %
                                (pre_dir, base_dir))

    # if path to the normalized values file does not exist
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

    weights = feather.read_dataframe(os.path.join(base_dir, weights_name))

    # ensure the norm vals columns and the FOV data contain valid indexes
    # ignoring metadata columns in the FOV data, the columns need to be in exactly
    # the same order across both datasets (normalized values and FOV values)
    pre_files = io_utils.list_files(preprocessed_path, substrs='.feather')
    norm_vals = feather.read_dataframe(os.path.join(base_dir, norm_vals_name))
    sample_fov = feather.read_dataframe(os.path.join(base_dir, pre_dir, pre_files[0]))

    cols_to_drop = ['fov', 'row_index', 'column_index']
    if 'segmentation_label' in sample_fov.columns.values:
        cols_to_drop.append('segmentation_label')

    sample_fov = sample_fov.drop(
        columns=cols_to_drop
    )
    misc_utils.verify_same_elements(
        enforce_order=True,
        norm_vals_columns=norm_vals.columns.values,
        pixel_data_columns=sample_fov.columns.values
    )

    # ensure the weights columns are valid indexes
    misc_utils.verify_same_elements(
        enforce_order=True,
        pixel_weights_columns=weights.columns.values,
        pixel_data_columns=sample_fov.columns.values
    )

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

    # compute average channel expression for each pixel SOM cluster
    # and the number of pixels per SOM cluster
    print("Computing average channel expression across pixel SOM clusters")
    pixel_channel_avg_som_cluster = compute_pixel_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        'pixel_som_cluster',
        cluster_dir,
        keep_count=True
    )

    # save pixel_channel_avg_som_cluster
    pixel_channel_avg_som_cluster.to_csv(
        os.path.join(base_dir, pc_chan_avg_som_cluster_name),
        index=False
    )


def pixel_consensus_cluster(fovs, channels, base_dir, max_k=20, cap=3,
                            cluster_dir='pixel_mat_clustered',
                            pc_chan_avg_som_cluster_name='pixel_channel_avg_som_cluster.csv',
                            pc_chan_avg_meta_cluster_name='pixel_channel_avg_meta_cluster.csv',
                            clust_to_meta_name='pixel_clust_to_meta.feather',
                            consensus_dir='pixel_mat_consensus', seed=42):
    """Run consensus clustering algorithm on pixel-level summed data across channels

    Saves data with consensus cluster labels to `consensus_dir`. Computes and saves the
    average channel expression across pixel meta clusters. Assigns meta cluster labels
    to the data stored in `pc_chan_avg_som_cluster_name`.

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
            Created by `cluster_pixels`
        pc_chan_avg_som_cluster_name (str):
            Name of file to save the channel-averaged results across all SOM clusters to
        pc_chan_avg_meta_cluster_name (str):
            Name of file to save the channel-averaged results across all meta clusters to
        consensus_dir (str):
            Name of directory to save the consensus clustered results (SOM and meta labels added)
        clust_to_meta_name (str):
            Name of file storing the SOM cluster to meta cluster mapping
        seed (int):
            The random seed to set for consensus clustering
    """

    # define the paths to the data
    clustered_path = os.path.join(base_dir, cluster_dir)
    som_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_som_cluster_name)
    consensus_path = os.path.join(base_dir, consensus_dir)
    clust_to_meta_path = os.path.join(base_dir, clust_to_meta_name)

    # if the path to the SOM clustered data doesn't exist
    if not os.path.exists(clustered_path):
        raise FileNotFoundError(
            'Cluster dir %s does not exist in base_dir %s' %
            (cluster_dir, base_dir)
        )

    # if the path to the average channel expression per SOM cluster doesn't exist
    if not os.path.exists(som_cluster_avg_path):
        raise FileNotFoundError(
            'Channel avg per SOM cluster file %s does not exist in base_dir %s' %
            (pc_chan_avg_som_cluster_name, base_dir)
        )

    # make consensus_dir if it doesn't exist
    if not os.path.exists(consensus_path):
        os.mkdir(consensus_path)

    # run the consensus clustering process
    process_args = ['Rscript', '/pixel_consensus_cluster.R', ','.join(fovs), ','.join(channels),
                    str(max_k), str(cap), clustered_path, som_cluster_avg_path, consensus_path,
                    clust_to_meta_path, str(seed)]

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

    # compute average channel expression for each pixel meta cluster
    # and the number of pixels per meta cluster
    print("Computing average channel expression across pixel meta clusters")
    pixel_channel_avg_meta_cluster = compute_pixel_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        'pixel_meta_cluster',
        consensus_dir,
        keep_count=True
    )

    # save pixel_channel_avg_meta_cluster
    pixel_channel_avg_meta_cluster.to_csv(
        os.path.join(base_dir, pc_chan_avg_meta_cluster_name),
        index=False
    )

    # read in the clust_to_meta_name file
    print("Mapping meta cluster values onto average channel expression across pixel SOM clusters")
    som_to_meta_data = feather.read_dataframe(
        os.path.join(base_dir, clust_to_meta_name)
    ).astype(np.int64)

    # read in the channel-averaged results across all pixel SOM clusters
    pixel_channel_avg_som_cluster = pd.read_csv(som_cluster_avg_path)

    # merge metacluster assignments in
    pixel_channel_avg_som_cluster = pd.merge_asof(
        pixel_channel_avg_som_cluster, som_to_meta_data, on='pixel_som_cluster'
    )

    # resave channel-averaged results across all pixel SOM clusters with metacluster assignments
    pixel_channel_avg_som_cluster.to_csv(
        som_cluster_avg_path,
        index=False
    )


def apply_pixel_meta_cluster_remapping(fovs, channels, base_dir,
                                       pixel_consensus_dir,
                                       pixel_remapped_name,
                                       pc_chan_avg_som_cluster_name,
                                       pc_chan_avg_meta_cluster_name):
    """Apply the meta cluster remapping to the data in `pixel_consensus_dir`.

    Resave the re-mapped consensus data to `pixel_consensus_dir` and re-runs the
    average channel expression per pixel meta cluster computation.

    Re-maps the pixel SOM clusters to meta clusters in `pc_chan_avg_som_cluster_name`.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directories
        pixel_consensus_dir (str):
            Name of directory with the pixel-level consensus data (SOM and meta labels added)
        pixel_remapped_name (str):
            Name of the file containing the pixel SOM clusters to their remapped meta clusters
        pc_chan_avg_som_cluster_name (str):
            Name of the file containing the channel-averaged results across all SOM clusters
        pc_chan_avg_meta_cluster_name (str):
            Name of the file containing the channel-averaged results across all meta clusters
    """

    # define the data paths
    pixel_consensus_path = os.path.join(base_dir, pixel_consensus_dir)
    pixel_remapped_path = os.path.join(base_dir, pixel_remapped_name)
    som_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_som_cluster_name)
    meta_cluster_avg_path = os.path.join(base_dir, pc_chan_avg_meta_cluster_name)

    # file path validation
    if not os.path.exists(pixel_consensus_path):
        raise FileNotFoundError('Pixel consensus dir %s does not exist in base_dir %s' %
                                (pixel_consensus_dir, base_dir))

    if not os.path.exists(pixel_remapped_path):
        raise FileNotFoundError('Pixel remapping file %s does not exist in base_dir %s' %
                                (pixel_remapped_name, base_dir))

    if not os.path.exists(som_cluster_avg_path):
        raise FileNotFoundError(
            'Channel average per SOM cluster file %s does not exist in base_dir %s' %
            (pc_chan_avg_meta_cluster_name, base_dir))

    if not os.path.exists(meta_cluster_avg_path):
        raise FileNotFoundError(
            'Channel average per meta cluster file %s does not exist in base_dir %s' %
            (pc_chan_avg_meta_cluster_name, base_dir))

    # read in the remapping
    pixel_remapped_data = pd.read_csv(pixel_remapped_path)

    # assert the correct columns are contained
    misc_utils.verify_same_elements(
        remapped_data_cols=pixel_remapped_data.columns.values,
        required_cols=['cluster', 'metacluster', 'mc_name']
    )

    # rename columns in pixel_remapped_data so it plays better with the existing
    # pixel_som_cluster and pixel_meta_cluster
    pixel_remapped_data = pixel_remapped_data.rename(
        {
            'cluster': 'pixel_som_cluster',
            'metacluster': 'pixel_meta_cluster',
            'mc_name': 'pixel_meta_cluster_rename'
        },
        axis=1
    )

    # create the mapping from pixel SOM to pixel meta cluster
    pixel_remapped_dict = dict(
        pixel_remapped_data[
            ['pixel_som_cluster', 'pixel_meta_cluster']
        ].values
    )

    # create the mapping from pixel meta cluster to renamed pixel meta cluster
    pixel_renamed_meta_dict = dict(
        pixel_remapped_data[
            ['pixel_meta_cluster', 'pixel_meta_cluster_rename']
        ].drop_duplicates().values
    )

    print("Using re-mapping scheme to re-label pixel meta clusters")
    for fov in fovs:
        # get the path to the fov
        fov_path = os.path.join(pixel_consensus_path, fov + '.feather')

        # read in the fov data with SOM and meta cluster labels
        fov_data = feather.read_dataframe(fov_path)

        # ensure that no SOM clusters are missing from the mapping
        misc_utils.verify_in_list(
            fov_som_labels=fov_data['pixel_som_cluster'],
            som_labels_in_mapping=list(pixel_remapped_dict.keys())
        )

        # assign the new meta cluster labels
        fov_data['pixel_meta_cluster'] = fov_data['pixel_som_cluster'].map(
            pixel_remapped_dict
        )

        # assign the renamed meta cluster names
        fov_data['pixel_meta_cluster_rename'] = fov_data['pixel_meta_cluster'].map(
            pixel_renamed_meta_dict
        )

        # resave the data with the new meta cluster lables
        feather.write_dataframe(fov_data, fov_path, compression='uncompressed')

    # re-compute average channel expression for each pixel meta cluster
    # and the number of pixels per meta cluster, add renamed meta cluster column in
    print("Re-computing average channel expression across pixel meta clusters")
    pixel_channel_avg_meta_cluster = compute_pixel_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        'pixel_meta_cluster',
        pixel_consensus_dir,
        keep_count=True
    )
    pixel_channel_avg_meta_cluster['pixel_meta_cluster_rename'] = \
        pixel_channel_avg_meta_cluster['pixel_meta_cluster'].map(pixel_renamed_meta_dict)

    # re-save the pixel channel average meta cluster table
    pixel_channel_avg_meta_cluster.to_csv(meta_cluster_avg_path, index=False)

    # re-assign pixel meta cluster labels back to the pixel channel average som cluster table
    print("Re-assigning meta cluster column in pixel SOM cluster average channel expression table")
    pixel_channel_avg_som_cluster = pd.read_csv(som_cluster_avg_path)

    pixel_channel_avg_som_cluster['pixel_meta_cluster'] = \
        pixel_channel_avg_som_cluster['pixel_som_cluster'].map(pixel_remapped_dict)

    pixel_channel_avg_som_cluster['pixel_meta_cluster_rename'] = \
        pixel_channel_avg_som_cluster['pixel_meta_cluster'].map(pixel_renamed_meta_dict)

    # re-save the pixel channel average som cluster table
    pixel_channel_avg_som_cluster.to_csv(som_cluster_avg_path, index=False)


def train_cell_som(fovs, channels, base_dir, pixel_consensus_dir, cell_table_name,
                   cluster_counts_name='cluster_counts.feather',
                   cluster_counts_norm_name='cluster_counts_norm.feather',
                   pixel_cluster_col='pixel_meta_cluster_rename',
                   pc_chan_avg_name='pc_chan_avg.csv',
                   weights_name='cell_weights.feather',
                   weighted_cell_channel_name='weighted_cell_channel.csv',
                   xdim=10, ydim=10, lr_start=0.05, lr_end=0.01, num_passes=1, seed=42):
    """Run the SOM training on the number of pixel/meta clusters in each cell of each fov

    Saves the weights to `base_dir/weights_name`. Computes and saves weighted
    channel expression per cell.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels used in pixel clustering
        base_dir (str):
            The path to the data directories
        pixel_consensus_dir (str):
            Name of directory with the pixel-level consensus data (SOM and meta labels added)
            Created by `pixel_consensus_cluster`
        cell_table_name (str):
            Name of the cell table, needs to be created with `Segment_Image_Data.ipynb`
        cluster_counts_name (str):
            Name of the file to save the number of pixel SOM/meta cluster counts for each cell
        cluster_counts_norm_name (str):
            Same as `cluster_counts_name`, except the cluster columns are normalized by
            `cell_size`
        pixel_cluster_col (str):
            Name of the column with the pixel SOM cluster assignments.
            Should be `'pixel_som_cluster'` or `'pixel_meta_cluster_rename'`.
        pc_chan_avg_name (str):
            Name of the file containing the average channel expression per pixel cluster.
            Which one chosen (SOM or meta averages) is set in the cell clustering notebook
            depending on the value of `pixel_cluster_col`.
        weights_name (str):
            The name of the file to save the weights to
        weighted_cell_channel_name (str):
            The name of the file to save the weighted channel expression table to
        xdim (int):
            The number of x nodes to use for the SOM
        ydim (int):
            The number of y nodes to use for the SOM
        lr_start (float):
            The start learning rate for the SOM, decays to `lr_end`
        lr_end (float):
            The end learning rate for the SOM, decays from `lr_start`
        num_passes (int):
            The number of training passes to make through the dataset
        seed (int):
            The random seed to set for training
    """

    # define the data paths
    cell_table_path = os.path.join(base_dir, cell_table_name)
    consensus_path = os.path.join(base_dir, pixel_consensus_dir)
    cluster_counts_path = os.path.join(base_dir, cluster_counts_name)
    cluster_counts_norm_path = os.path.join(base_dir, cluster_counts_norm_name)
    weights_path = os.path.join(base_dir, weights_name)

    # if the cell table path does not exist
    if not os.path.exists(cell_table_path):
        raise FileNotFoundError('Cell table %s does not exist in base_dir %s' %
                                (cell_table_name, base_dir))

    # if the pixel data with the SOM and meta labels path does not exist
    if not os.path.exists(consensus_path):
        raise FileNotFoundError('Pixel consensus dir %s does not exist in base_dir %s' %
                                (consensus_path, base_dir))

    # verify the cluster_col provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster_rename']
    )

    # generate matrices with each fov/cell label pair with their pixel SOM/meta cluster counts
    # NOTE: a normalized and an un-normalized matrix (by cell size) will be created
    # NOTE: we'll need the un-normalized matrix to compute weighted channel average
    # but the normalized matrix will be used to train, SOM cluster, and consensus cluster
    print("Counting the number of pixel SOM/meta cluster counts for each fov/cell pair")
    cluster_counts, cluster_counts_norm = create_c2pc_data(
        fovs, consensus_path, cell_table_path, pixel_cluster_col
    )

    # write the created matrices
    feather.write_dataframe(cluster_counts,
                            cluster_counts_path,
                            compression='uncompressed')
    feather.write_dataframe(cluster_counts_norm,
                            cluster_counts_norm_path,
                            compression='uncompressed')

    # run the SOM training process
    process_args = ['Rscript', '/create_cell_som.R', ','.join(fovs), str(xdim), str(ydim),
                    str(lr_start), str(lr_end), str(num_passes), cluster_counts_norm_path,
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

    # read in the pixel channel averages table
    print("Computing the weighted channel expression per cell")
    pixel_cluster_channel_avg = pd.read_csv(os.path.join(base_dir, pc_chan_avg_name))

    # compute the weighted channel table
    weighted_cell_channel = compute_p2c_weighted_channel_avg(
        pixel_cluster_channel_avg, channels, cluster_counts,
        fovs=fovs, pixel_cluster_col=pixel_cluster_col
    )

    # save the weighted channel table
    weighted_cell_channel.to_csv(
        os.path.join(base_dir, weighted_cell_channel_name),
        index=False
    )


def cluster_cells(base_dir, cluster_counts_norm_name='cluster_counts_norm.feather',
                  weights_name='cell_weights.feather',
                  cell_cluster_name='cell_mat_clustered.feather',
                  pixel_cluster_col_prefix='pixel_meta_cluster_rename',
                  cell_som_cluster_count_avgs_name='cell_som_cluster_count_avgs.csv'):
    """Uses trained weights to assign cluster labels on full cell data

    Saves data with cluster labels to `cell_cluster_name`. Computes and saves the average number
    of pixel SOM/meta clusters per cell SOM cluster.

    Args:
        base_dir (str):
            The path to the data directory
        cluster_counts_norm_name (str):
            Name of the file with the number of pixel SOM/meta cluster counts of each cell,
            normalized by `cell_size`
        weights_name (str):
            The name of the weights file, created by `train_cell_som`
        cell_cluster_name (str):
            Name of the file to save the cell data with cell SOM cluster labels
        pixel_cluster_col_prefix (str):
            The name of the prefixes of each of the pixel SOM/meta columns
            Should be `'pixel_som_cluster'` or `'pixel_meta_cluster_rename'`.
        cell_som_cluster_count_avgs_name (str):
            The name of the file to write the clustered data
    """

    # define the paths to the data
    cluster_counts_norm_path = os.path.join(base_dir, cluster_counts_norm_name)
    weights_path = os.path.join(base_dir, weights_name)
    cell_cluster_path = os.path.join(base_dir, cell_cluster_name)

    # if the path to the normalized pixel cluster counts per cell doesn't exist
    if not os.path.exists(cluster_counts_norm_path):
        raise FileNotFoundError(
            'Normalized pixel cluster counts per cell file %s does not exist in base_dir %s' %
            (cluster_counts_norm_name, base_dir)
        )

    # if the path to the weights file does not exist
    if not os.path.exists(weights_path):
        raise FileNotFoundError('Weights file %s does not exist in base_dir %s' %
                                (weights_name, base_dir))

    # verify the pixel_cluster_col_prefix provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col_prefix],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster_rename']
    )

    # ensure the weights columns are valid indexes, do so by ensuring
    # the cluster_counts_norm and weights columns are the same
    # minus the metadata columns that appear in cluster_counts_norm
    cluster_counts_norm = feather.read_dataframe(cluster_counts_norm_path)
    weights = feather.read_dataframe(os.path.join(base_dir, weights_name))
    cluster_counts_norm = cluster_counts_norm.drop(
        columns=['fov', 'segmentation_label', 'cell_size']
    )
    misc_utils.verify_same_elements(
        enforce_order=True,
        cluster_counts_norm_columns=cluster_counts_norm.columns.values,
        cell_weights_columns=weights.columns.values
    )

    # run the trained SOM on the dataset, assigning clusters
    process_args = ['Rscript', '/run_cell_som.R', cluster_counts_norm_path,
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

    # compute the average pixel SOM/meta counts per cell SOM cluster
    print("Computing the average number of pixel SOM/meta cluster counts per cell SOM cluster")
    cell_som_cluster_avgs_and_counts = compute_cell_cluster_count_avg(
        cell_cluster_path,
        pixel_cluster_col_prefix,
        'cell_som_cluster',
        keep_count=True
    )

    # save the average pixel SOM/meta counts per cell SOM cluster
    cell_som_cluster_avgs_and_counts.to_csv(
        os.path.join(base_dir, cell_som_cluster_count_avgs_name),
        index=False
    )


def cell_consensus_cluster(fovs, channels, base_dir, pixel_cluster_col, max_k=20, cap=3,
                           cell_cluster_name='cell_mat_clustered.feather',
                           cell_som_cluster_count_avgs_name='cell_som_cluster_avgs.csv',
                           cell_meta_cluster_count_avgs_name='cell_meta_cluster_avgs.csv',
                           weighted_cell_channel_name='weighted_cell_channel.csv',
                           cell_som_cluster_channel_avg_name='cell_som_cluster_channel_avg.csv',
                           cell_meta_cluster_channel_avg_name='cell_meta_cluster_channel_avg.csv',
                           clust_to_meta_name='cell_clust_to_meta.feather',
                           cell_consensus_name='cell_mat_consensus.feather', seed=42):
    """Run consensus clustering algorithm on cell-level data averaged across each cell SOM cluster

    Saves data with consensus cluster labels to cell_consensus_name. Computes and saves the
    average number of pixel SOM/meta clusters per cell meta cluster. Assigns meta cluster labels
    to the data stored in `cell_som_cluster_count_avgs_name`.

    Computes and saves the average weighted cell channel expression per cell SOM and meta cluster.

    Args:
        fovs (list):
            The list of fovs to subset on (from pixel clustering)
        channels (list):
            The list of channels to subset on (from pixel clustering)
        base_dir (str):
            The path to the data directory
        pixel_cluster_col (str):
            Name of the column used to generate the pixel SOM/meta cluster counts.
            Should be `'pixel_som_cluster'` or `'pixel_meta_cluster_rename'`.
        max_k (int):
            The number of consensus clusters
        cap (int):
            z-score cap to use when hierarchical clustering
        cell_cluster_name (str):
            Name of the file containing the cell data with cell SOM cluster labels
            Created by cluster_cells
        cell_som_cluster_count_avgs_name (str):
            The average number of pixel SOM/meta clusters per cell SOM cluster.
            Used to run consensus clustering on.
        cell_meta_cluster_count_avgs_name (str):
            Same as above except for cell meta clusters
        weighted_cell_channel_name (str):
            The name of the file containing the weighted channel expression table
        cell_som_cluster_channel_avg_name (str):
            The name of the file to save the average weighted channel expression
            per cell SOM cluster
        cell_meta_cluster_channel_avg_name (str):
            Same as above except for cell meta clusters
        clust_to_meta_name (str):
            Name of file storing the SOM cluster to meta cluster mapping
        cell_consensus_name (str):
            Name of file to save the consensus clustered results (SOM and meta labels added)
        seed (int):
            The random seed to set for consensus clustering
    """

    # define the paths to the data
    clustered_path = os.path.join(base_dir, cell_cluster_name)
    som_cluster_counts_avg_path = os.path.join(base_dir, cell_som_cluster_count_avgs_name)
    consensus_path = os.path.join(base_dir, cell_consensus_name)
    weighted_channel_path = os.path.join(base_dir, weighted_cell_channel_name)
    clust_to_meta_path = os.path.join(base_dir, clust_to_meta_name)

    # if the path to the SOM clustered data doesn't exist
    if not os.path.exists(clustered_path):
        raise FileNotFoundError(
            'Cell SOM cluster data file %s does not exist in base_dir %s' %
            (cell_cluster_name, base_dir)
        )

    # if the path to the average pixel cluster counts per cell cluster doesn't exist
    if not os.path.exists(som_cluster_counts_avg_path):
        raise FileNotFoundError(
            'Average pix clust count per cell SOM cluster file %s does not exist in base_dir %s' %
            (cell_som_cluster_count_avgs_name, base_dir)
        )

    # if the path to the weighted channel data doesn't exist
    if not os.path.exists(weighted_channel_path):
        raise FileNotFoundError(
            'Weighted channel table %s does not exist in base_dir %s' %
            (weighted_cell_channel_name, base_dir)
        )

    # verify the pixel_cluster_col provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster_rename']
    )

    # run the consensus clustering process
    process_args = ['Rscript', '/cell_consensus_cluster.R', pixel_cluster_col,
                    str(max_k), str(cap), clustered_path,
                    som_cluster_counts_avg_path, consensus_path, clust_to_meta_path, str(seed)]

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

    # compute the average pixel SOM/meta counts per cell meta cluster
    print("Compute the average number of pixel SOM/meta cluster counts per cell meta cluster")
    cell_meta_cluster_avgs_and_counts = compute_cell_cluster_count_avg(
        consensus_path,
        pixel_cluster_col,
        'cell_meta_cluster',
        keep_count=True
    )

    # save the average pixel SOM/meta counts per cell meta cluster
    cell_meta_cluster_avgs_and_counts.to_csv(
        os.path.join(base_dir, cell_meta_cluster_count_avgs_name),
        index=False
    )

    # read in the clust_to_meta_name file
    print(
        "Mapping meta cluster values onto average number of pixel SOM/meta cluster counts"
        "across cell SOM clusters"
    )
    som_to_meta_data = feather.read_dataframe(
        os.path.join(base_dir, clust_to_meta_name)
    ).astype(np.int64)

    # read in the average number of pixel/SOM clusters across all cell SOM clusters
    cell_som_cluster_avgs_and_counts = pd.read_csv(som_cluster_counts_avg_path)

    # merge metacluster assignments in
    cell_som_cluster_avgs_and_counts = pd.merge_asof(
        cell_som_cluster_avgs_and_counts, som_to_meta_data, on='cell_som_cluster'
    )

    # resave average number of pixel/SOM clusters across all cell SOM clusters
    # with metacluster assignments
    cell_som_cluster_avgs_and_counts.to_csv(
        som_cluster_counts_avg_path,
        index=False
    )

    # compute the weighted channel average expression per cell SOM cluster
    print("Compute average weighted channel expression across cell SOM clusters")
    cell_som_cluster_channel_avg = compute_cell_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        weighted_cell_channel_name,
        cell_consensus_name,
        'cell_som_cluster'
    )

    # merge metacluster assignments into cell_som_cluster_channel_avg
    print(
        "Mapping meta cluster values onto average weighted channel expression"
        "across cell SOM clusters"
    )
    cell_som_cluster_channel_avg = pd.merge_asof(
        cell_som_cluster_channel_avg, som_to_meta_data, on='cell_som_cluster'
    )

    # save the weighted channel average expression per cell cluster
    cell_som_cluster_channel_avg.to_csv(
        os.path.join(base_dir, cell_som_cluster_channel_avg_name),
        index=False
    )

    # compute the weighted channel average expression per cell meta cluster
    print("Compute average weighted channel expression across cell meta clusters")
    cell_meta_cluster_channel_avg = compute_cell_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        weighted_cell_channel_name,
        cell_consensus_name,
        'cell_meta_cluster'
    )

    # save the weighted channel average expression per cell cluster
    cell_meta_cluster_channel_avg.to_csv(
        os.path.join(base_dir, cell_meta_cluster_channel_avg_name),
        index=False
    )


def apply_cell_meta_cluster_remapping(fovs, channels, base_dir, cell_consensus_name,
                                      cell_remapped_name,
                                      pixel_cluster_col,
                                      cell_som_cluster_count_avgs_name,
                                      cell_meta_cluster_count_avgs_name,
                                      weighted_cell_channel_name,
                                      cell_som_cluster_channel_avg_name,
                                      cell_meta_cluster_channel_avg_name):
    """Apply the meta cluster remapping to the data in `cell_consensus_name`.

    Resave the re-mapped consensus data to `cell_consensus_name` and re-runs the
    weighted channel expression and average pixel SOM/meta cluster counts per cell
    SOM cluster.

    Re-maps the pixel SOM clusters to meta clusters in `cell_som_cluster_count_avgs_name` and
    `cell_som_cluster_channel_avg_name`

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directory
        cell_consensus_name (str):
            Name of file with the cell consensus clustered results (both cell SOM and meta labels)
        cell_remapped_name (str):
            Name of the file containing the cell SOM clusters to their remapped meta clusters
        pixel_cluster_col (str):
            Name of the column used to generate the pixel SOM/meta cluster counts.
            Should be `'pixel_som_cluster'` or `'pixel_meta_cluster_rename'`.
        cell_som_cluster_count_avgs_name (str):
            The average number of pixel SOM/meta clusters per cell SOM cluster
        cell_meta_cluster_count_avgs_name (str):
            Same as above except for cell meta clusters
        weighted_cell_channel_name (str):
            The name of the file containing the weighted channel expression table
        cell_som_cluster_channel_avg_name (str):
            The name of the file to save the average weighted channel expression
            per cell SOM cluster
        cell_meta_cluster_channel_avg_name (str):
            Same as above except for cell meta clusters
    """

    # define the data paths
    cell_consensus_path = os.path.join(base_dir, cell_consensus_name)
    cell_remapped_path = os.path.join(base_dir, cell_remapped_name)
    som_cluster_counts_avgs_path = os.path.join(base_dir, cell_som_cluster_count_avgs_name)
    meta_cluster_counts_avgs_path = os.path.join(base_dir, cell_meta_cluster_count_avgs_name)
    weighted_channel_path = os.path.join(base_dir, weighted_cell_channel_name)
    som_cluster_channel_avgs_path = os.path.join(base_dir, cell_som_cluster_channel_avg_name)
    meta_cluster_channel_avgs_path = os.path.join(base_dir, cell_meta_cluster_channel_avg_name)

    # file path validation
    if not os.path.exists(cell_consensus_path):
        raise FileNotFoundError('Cell consensus file %s does not exist in base_dir %s' %
                                (cell_consensus_name, base_dir))

    if not os.path.exists(cell_remapped_path):
        raise FileNotFoundError('Cell remapping file %s does not exist in base_dir %s' %
                                (cell_remapped_name, base_dir))

    if not os.path.exists(som_cluster_counts_avgs_path):
        raise FileNotFoundError(
            'Average pix clust count per cell SOM cluster file %s does not exist in base_dir %s' %
            (cell_som_cluster_count_avgs_name, base_dir)
        )

    if not os.path.exists(meta_cluster_counts_avgs_path):
        raise FileNotFoundError(
            'Average pix clust count per cell meta cluster file %s does not exist in base_dir %s' %
            (cell_meta_cluster_count_avgs_name, base_dir)
        )

    if not os.path.exists(weighted_channel_path):
        raise FileNotFoundError('Weighted channel table %s does not exist in base_dir %s' %
                                (weighted_cell_channel_name, base_dir))

    if not os.path.exists(som_cluster_channel_avgs_path):
        raise FileNotFoundError(
            'Average weighted chan per cell SOM cluster file %s does not exist in base_dir %s' %
            (cell_som_cluster_channel_avg_name, base_dir)
        )

    if not os.path.exists(meta_cluster_channel_avgs_path):
        raise FileNotFoundError(
            'Average weighted chan per cell meta cluster file %s does not exist in base_dir %s' %
            (cell_meta_cluster_channel_avg_name, base_dir)
        )

    # verify the pixel_cluster_col provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster_rename']
    )

    # read in the remapping
    cell_remapped_data = pd.read_csv(cell_remapped_path)

    # assert the correct columns are contained
    misc_utils.verify_same_elements(
        remapped_data_cols=cell_remapped_data.columns.values,
        required_cols=['cluster', 'metacluster', 'mc_name']
    )

    # rename columns in pixel_remapped_data so it plays better with the existing
    # cell_som_cluster and cell_meta_cluster
    cell_remapped_data = cell_remapped_data.rename(
        {
            'cluster': 'cell_som_cluster',
            'metacluster': 'cell_meta_cluster',
            'mc_name': 'cell_meta_cluster_rename'
        },
        axis=1
    )

    # create the mapping from cell SOM to cell meta cluster
    cell_remapped_dict = dict(
        cell_remapped_data[
            ['cell_som_cluster', 'cell_meta_cluster']
        ].values
    )

    # create the mapping from cell meta cluster to cell renamed meta cluster
    cell_renamed_meta_dict = dict(
        cell_remapped_data[
            ['cell_meta_cluster', 'cell_meta_cluster_rename']
        ].drop_duplicates().values
    )

    # load the cell consensus data in
    print("Using re-mapping scheme to re-label cell meta clusters")
    cell_consensus_data = feather.read_dataframe(cell_consensus_path)

    # ensure that no SOM clusters are missing from the mapping
    misc_utils.verify_in_list(
        fov_som_labels=cell_consensus_data['cell_som_cluster'],
        som_labels_in_mapping=list(cell_remapped_dict.keys())
    )

    # assign the new meta cluster labels
    cell_consensus_data['cell_meta_cluster'] = \
        cell_consensus_data['cell_som_cluster'].map(cell_remapped_dict)

    # assign the new renamed meta cluster names
    # assign the new meta cluster labels
    cell_consensus_data['cell_meta_cluster_rename'] = \
        cell_consensus_data['cell_meta_cluster'].map(cell_renamed_meta_dict)

    # resave the data with the new meta cluster lables
    feather.write_dataframe(cell_consensus_data, cell_consensus_path, compression='uncompressed')

    # re-compute the average number of pixel SOM/meta clusters per cell meta cluster
    # add renamed meta cluster in
    print("Re-compute pixel SOM/meta cluster count per cell meta cluster")
    cell_meta_cluster_avgs_and_counts = compute_cell_cluster_count_avg(
        cell_consensus_path,
        pixel_cluster_col,
        'cell_meta_cluster',
        keep_count=True
    )

    cell_meta_cluster_avgs_and_counts['cell_meta_cluster_rename'] = \
        cell_meta_cluster_avgs_and_counts['cell_meta_cluster'].map(cell_renamed_meta_dict)

    # re-save the average number of pixel SOM/meta clusters per cell meta cluster
    cell_meta_cluster_avgs_and_counts.to_csv(
        meta_cluster_counts_avgs_path,
        index=False
    )

    # re-compute the weighted channel average expression per cell meta cluster
    # add renamed meta cluster in
    print("Re-compute average weighted channel expression across cell meta clusters")
    cell_meta_cluster_channel_avg = compute_cell_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        weighted_cell_channel_name,
        cell_consensus_name,
        'cell_meta_cluster'
    )

    cell_meta_cluster_channel_avg['cell_meta_cluster_rename'] = \
        cell_meta_cluster_channel_avg['cell_meta_cluster'].map(cell_renamed_meta_dict)

    # re-save the weighted channel average expression per cell cluster
    cell_meta_cluster_channel_avg.to_csv(
        meta_cluster_channel_avgs_path,
        index=False
    )

    # re-assign cell meta cluster labels back to the average pixel cluster counts
    # per cell SOM cluster table
    print("Re-assigning meta cluster column in cell SOM cluster average pixel cluster counts data")
    cell_som_cluster_avgs_and_counts = pd.read_csv(som_cluster_counts_avgs_path)

    cell_som_cluster_avgs_and_counts['cell_meta_cluster'] = \
        cell_som_cluster_avgs_and_counts['cell_som_cluster'].map(cell_remapped_dict)

    cell_som_cluster_avgs_and_counts['cell_meta_cluster_rename'] = \
        cell_som_cluster_avgs_and_counts['cell_meta_cluster'].map(cell_renamed_meta_dict)

    # re-save the cell SOM cluster average pixel cluster counts table
    cell_som_cluster_avgs_and_counts.to_csv(som_cluster_counts_avgs_path, index=False)

    # re-assign cell meta cluster labels back to the average weighted channel expression
    # per cell SOM cluster table
    print("Re-assigning meta cluster column in cell SOM cluster average weighted channel data")
    cell_som_cluster_channel_avg = pd.read_csv(som_cluster_channel_avgs_path)

    cell_som_cluster_channel_avg['cell_meta_cluster'] = \
        cell_som_cluster_channel_avg['cell_som_cluster'].map(cell_remapped_dict)

    cell_som_cluster_channel_avg['cell_meta_cluster_rename'] = \
        cell_som_cluster_channel_avg['cell_meta_cluster'].map(cell_renamed_meta_dict)

    # re-save the cell SOM cluster average pixel cluster counts table
    cell_som_cluster_channel_avg.to_csv(som_cluster_channel_avgs_path, index=False)


def generate_meta_cluster_colormap_dict(meta_cluster_remap_path, cmap):
    """Returns a compact version of the colormap used in the interactive reclustering processes.

    Generate a separate one for the raw meta cluster labels and the renamed meta cluster labels.

    Used in the pixel and cell meta cluster overlays, as well as the
    average weighted channel expression heatmaps for cell clustering

    Args:
        meta_cluster_remap_path (str):
            Path to the file storing the mapping from SOM to meta clusters (raw and renamed)
        cmap (matplotlib.colors.ListedColormap):
            The colormap generated by the interactive reclustering process

    Returns:
        tuple:

        - A `dict` containing the raw meta cluster labels mapped to their respective colors
        - A `dict` containing the renamed meta cluster labels mapped to their respective colors
    """

    # file path validation
    if not os.path.exists(meta_cluster_remap_path):
        raise FileNotFoundError('Remapping path %s does not exist' %
                                meta_cluster_remap_path)

    # read the remapping
    remapping = pd.read_csv(meta_cluster_remap_path)

    # assert the correct columns are contained
    misc_utils.verify_same_elements(
        remapping_cols=remapping.columns.values,
        required_cols=['cluster', 'metacluster', 'mc_name']
    )

    # define the raw meta cluster colormap
    # NOTE: colormaps returned by interactive reclustering are zero-indexed
    # need to subtract 1 to account for that
    raw_colormap = {
        i: cmap(i - 1) for i in np.unique(remapping['metacluster'])
    }

    # define the renamed meta cluster colormap
    meta_id_to_name = dict(zip(remapping['metacluster'], remapping['mc_name']))
    renamed_colormap = {
        meta_id_to_name[meta_id]: meta_id_color
        for (meta_id, meta_id_color) in raw_colormap.items()
    }

    return raw_colormap, renamed_colormap


def generate_weighted_channel_avg_heatmap(cell_cluster_channel_avg_path, cell_cluster_col,
                                          channels, raw_cmap, renamed_cmap,
                                          center_val=0, min_val=-3, max_val=3):
    """Generates a z-scored heatmap of the average weighted channel expression per cell cluster

    Args:
        cell_cluster_channel_avg_path (str):
            Path to the file containing the average weighted channel expression per cell cluster
        cell_cluster_col (str):
            The name of the cell cluster col,
            needs to be either 'cell_som_cluster' or 'cell_meta_cluster_rename'
        channels (str):
            The list of channels to visualize
        raw_cmap (dict):
            Maps the raw meta cluster labels to their respective colors,
            created by `generate_meta_cluster_colormap_dict`
        renamed_cmap (dict):
            Maps the renamed meta cluster labels to their respective colors,
            created by `generate_meta_cluster_colormap_dict`
        center_val (float):
            value at which to center the heatmap
        min_val (float):
            minimum value the heatmap should take
        max_val (float):
            maximum value the heatmap should take
    """

    # file path validation
    if not os.path.exists(cell_cluster_channel_avg_path):
        raise FileNotFoundError('Channel average path %s does not exist' %
                                cell_cluster_channel_avg_path)

    # verify the cell_cluster_col provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[cell_cluster_col],
        valid_cluster_cols=['cell_som_cluster', 'cell_meta_cluster_rename']
    )

    # read the channel average path
    cell_cluster_channel_avgs = pd.read_csv(cell_cluster_channel_avg_path)

    # assert the channels provided are valid
    misc_utils.verify_in_list(
        provided_channels=channels,
        channel_avg_cols=cell_cluster_channel_avgs.columns.values
    )

    # sort the data by the meta cluster value
    # this ensures the meta clusters are grouped together when the colormap is displayed
    cell_cluster_channel_avgs = cell_cluster_channel_avgs.sort_values(
        by='cell_meta_cluster_rename'
    )

    # map raw_cmap onto cell_cluster_channel_avgs for the heatmap to display the side color bar
    meta_cluster_index = cell_cluster_channel_avgs[cell_cluster_col].values
    meta_cluster_mapping = pd.Series(
        cell_cluster_channel_avgs['cell_meta_cluster_rename']
    ).map(renamed_cmap)
    meta_cluster_mapping.index = meta_cluster_index

    # draw the heatmap
    visualize.draw_heatmap(
        data=stats.zscore(cell_cluster_channel_avgs[channels].values),
        x_labels=cell_cluster_channel_avgs[cell_cluster_col],
        y_labels=channels,
        center_val=center_val,
        min_val=min_val,
        max_val=max_val,
        cbar_ticks=np.arange(-3, 4),
        row_colors=meta_cluster_mapping,
        row_cluster=False,
        left_start=0.0,
        right_start=0.85,
        w_spacing=0.2,
        colormap='vlag'
    )

    # add the legend
    handles = [patches.Patch(facecolor=raw_cmap[mc]) for mc in raw_cmap]
    _ = plt.legend(
        handles,
        renamed_cmap,
        title='Meta cluster',
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc='upper right'
    )
