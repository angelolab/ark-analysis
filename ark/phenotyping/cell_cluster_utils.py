from functools import partial
import multiprocessing
import os
import json
import subprocess
import warnings

import feather
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid
import re
import scipy.ndimage as ndimage
import scipy.stats as stats
from shutil import rmtree
from skimage.io import imread, imsave
import xarray as xr

from ark.analysis import visualize
import ark.settings as settings
from ark.utils import io_utils
from ark.utils import load_utils
from ark.utils import misc_utils


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


def create_c2pc_data(fovs, pixel_data_path,
                     cell_table_path, pixel_cluster_col='pixel_meta_cluster_rename'):
    """Create a matrix with each fov-cell label pair and their SOM pixel/meta cluster counts

    Args:
        fovs (list):
            The list of fovs to subset on
        pixel_data_path (str):
            Path to directory with the pixel data with SOM and meta labels attached.
            Created by `pixel_consensus_cluster`.
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
            os.path.join(pixel_data_path, fov + '.feather')
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
