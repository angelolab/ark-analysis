import os

import feather
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from alpineer import io_utils, misc_utils

from ark.analysis import visualize
from ark.phenotyping import cluster_helpers


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
    else:
        # verify that the fovs provided are valid
        misc_utils.verify_in_list(
            provided_fovs=fovs,
            dataset_fovs=cell_counts['fov'].unique()
        )

    # verify the pixel_cluster_col provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=pixel_cluster_col,
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
    if np.issubdtype(pixel_channel_avg[pixel_cluster_col].dtype, np.integer):
        pixel_channel_avg[pixel_cluster_col] = pixel_channel_avg[pixel_cluster_col].astype(str)

    # sort the pixel channel average by pixel cluster col for standardization
    # needed because the cell_counts_clusters columns are sorted by increasing pixel cluster
    pixel_channel_avg_sorted = pixel_channel_avg.sort_values(by=pixel_cluster_col)

    # retrieve the pixel SOM clusters represented in the cell counts table
    cell_counts_cluster_ids = [
        x.replace(pixel_cluster_col + '_', '') for x in cell_counts_clusters.columns.values
    ]

    # subset pixel channel cluster IDs on just the cell counts cluster IDs contained
    pixel_channel_avg_sorted = pixel_channel_avg_sorted[
        pixel_channel_avg_sorted[pixel_cluster_col].isin(cell_counts_cluster_ids)
    ]

    # retrieve the pixel cluster ids
    pixel_channel_cluster_ids = pixel_channel_avg_sorted[pixel_cluster_col].values

    # extra sanity checking, the matrix multiplication will fail otherwise
    # this should never fail, just as an added protection
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


def compute_cell_cluster_weighted_channel_avg(fovs, channels, base_dir,
                                              weighted_cell_channel_name,
                                              cell_cluster_data,
                                              cell_cluster_col='cell_meta_cluster'):
    """Computes the average weighted marker expression for each cell cluster

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directory
        weighted_cell_channel_name (str):
            The name of the weighted cell table, created in `3_Pixie_Cluster_Cells.ipynb`
        cell_cluster_data (pandas.DataFrame):
            Name of the file containing the cell data with cluster labels
        cell_cluster_col (str):
            Whether to aggregate by cell SOM or meta labels
            Needs to be either 'cell_som_cluster', or 'cell_meta_cluster'

    Returns:
        pandas.DataFrame:
            Each cell cluster mapped to the average expression for each marker
    """

    weighted_cell_channel_name_path: str = os.path.join(base_dir, weighted_cell_channel_name)

    # verify the cell table actually exists
    io_utils.validate_paths([weighted_cell_channel_name_path])

    # verify the cell cluster col specified is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[cell_cluster_col],
        valid_cluster_cols=['cell_som_cluster', 'cell_meta_cluster']
    )

    # read the weighted cell channel table in
    cell_table = feather.read_dataframe(weighted_cell_channel_name_path)

    # subset on only the fovs the user has specified
    cell_table = cell_table[cell_table['fov'].isin(fovs)]

    # need to ensure that both cell_table and cluster_data have FOVs and segmentation_labels sorted
    # in the same order, this can be done by simply sorting by fov and segmentation_label for both
    cell_table = cell_table.sort_values(
        by=['fov', 'segmentation_label']
    ).reset_index(drop=True)
    cell_cluster_data = cell_cluster_data.sort_values(
        by=['fov', 'segmentation_label']
    ).reset_index(drop=True)

    # add an extra check to ensure that the FOVs and segmentation labels are in the same order
    misc_utils.verify_same_elements(
        enforce_order=True,
        cell_table_fovs=list(cell_table['fov']),
        cluster_data_fovs=list(cell_cluster_data['fov'])
    )
    misc_utils.verify_same_elements(
        enforce_order=True,
        cell_table_labels=list(cell_table['segmentation_label']),
        cluster_data_labels=list(cell_cluster_data['segmentation_label'])
    )

    # assign the cluster labels to cell_table
    cell_table[cell_cluster_col] = cell_cluster_data[cell_cluster_col]

    # subset the cell table by just the desired channels and the cell_cluster_col
    cell_table = cell_table[channels + [cell_cluster_col]]

    # compute the mean channel expression across each cell cluster
    channel_avgs = cell_table.groupby(cell_cluster_col).mean().reset_index()

    return channel_avgs


def generate_wc_avg_files(fovs, channels, base_dir, cell_cc, cell_som_input_data,
                          weighted_cell_channel_name='weighted_cell_channel.feather',
                          cell_som_cluster_channel_avg_name='cell_som_cluster_channel_avg.csv',
                          cell_meta_cluster_channel_avg_name='cell_meta_cluster_channel_avg.csv',
                          overwrite=False):
    """Generate the weighted channel average files per cell SOM and meta clusters.

    When running cell clustering with pixel clusters generated from Pixie, the counts of each
    pixel cluster per cell is computed. These are multiplied by the average expression profile of
    each pixel cluster to determine weighted channel average. This computation is averaged by both
    cell SOM and meta cluster.

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directory
        cell_cc (cluster_helpers.PixieConsensusCluster):
            The consensus cluster object containing the SOM to meta mapping
        cell_som_input_data (str):
            The input data used for SOM training. For weighted channel averaging, it should
            contain the number of pixel SOM/meta cluster counts of each cell,
            normalized by `cell_size`.
        weighted_cell_channel_name (str):
            The name of the file containing the weighted channel expression table
        cell_som_cluster_channel_avg_name (str):
            The name of the file to save the average weighted channel expression
            per cell SOM cluster
        cell_meta_cluster_channel_avg_name (str):
            Same as above except for cell meta clusters
        overwrite (bool):
            If set, regenerate average weighted channel expression for SOM and meta clusters
    """
    # define the paths to the data
    weighted_channel_path = os.path.join(base_dir, weighted_cell_channel_name)
    som_cluster_channel_avg_path = os.path.join(base_dir, cell_som_cluster_channel_avg_name)
    meta_cluster_channel_avg_path = os.path.join(base_dir, cell_meta_cluster_channel_avg_name)

    # check paths
    io_utils.validate_paths([weighted_channel_path])

    # if the weighted channel average files exist, skip
    if os.path.exists(som_cluster_channel_avg_path) and \
       os.path.exists(meta_cluster_channel_avg_path):
        if not overwrite:
            print("Already generated average weighted channel expression files, skipping")
            return

        print("Overwrite flag set, regenerating average weighted channel expression files")

    print("Compute average weighted channel expression across cell SOM clusters")
    cell_som_cluster_channel_avg = compute_cell_cluster_weighted_channel_avg(
        fovs,
        channels,
        base_dir,
        weighted_cell_channel_name,
        cell_som_input_data,
        'cell_som_cluster'
    )

    # merge metacluster assignments into cell_som_cluster_channel_avg
    print(
        "Mapping meta cluster values onto average weighted channel expression"
        "across cell SOM clusters"
    )
    cell_som_cluster_channel_avg = pd.merge_asof(
        cell_som_cluster_channel_avg, cell_cc.mapping, on='cell_som_cluster'
    )

    # save the weighted channel average expression per cell cluster
    cell_som_cluster_channel_avg.to_csv(
        som_cluster_channel_avg_path,
        index=False
    )

    # compute the weighted channel average expression per cell meta cluster
    print("Compute average weighted channel expression across cell meta clusters")
    cell_meta_cluster_channel_avg = compute_cell_cluster_weighted_channel_avg(
        fovs,
        channels,
        base_dir,
        weighted_cell_channel_name,
        cell_som_input_data,
        'cell_meta_cluster'
    )

    # save the weighted channel average expression per cell cluster
    cell_meta_cluster_channel_avg.to_csv(
        meta_cluster_channel_avg_path,
        index=False
    )


def generate_remap_avg_wc_files(fovs, channels, base_dir, cell_som_input_data,
                                cell_remapped_name, weighted_cell_channel_name,
                                cell_som_cluster_channel_avg_name,
                                cell_meta_cluster_channel_avg_name):
    """Apply the cell cluster remapping to the average weighted channel files

    Args:
        fovs (list):
            The list of fovs to subset on
        channels (list):
            The list of channels to subset on
        base_dir (str):
            The path to the data directory
        cell_som_input_data (pandas.DataFrame):
            The input data used for SOM training. For weighted channel averaging, this should
            contain the number of pixel SOM/meta cluster counts of each cell,
            normalized by `cell_size`.
        cell_remapped_name (str):
            Name of the file containing the cell SOM clusters to their remapped meta clusters
        weighted_cell_channel_name (str):
            The name of the file containing the weighted channel expression table
        cell_som_cluster_channel_avg_name (str):
            The name of the file to save the average weighted channel expression
            per cell SOM cluster
        cell_meta_cluster_channel_avg_name (str):
            Same as above except for cell meta clusters
    """
    # define the data paths
    cell_remapped_path = os.path.join(base_dir, cell_remapped_name)
    weighted_cell_channel_path = os.path.join(base_dir, weighted_cell_channel_name)
    som_cluster_channel_avg_path = os.path.join(base_dir, cell_som_cluster_channel_avg_name)
    meta_cluster_channel_avg_path = os.path.join(base_dir, cell_meta_cluster_channel_avg_name)

    # file path validation
    io_utils.validate_paths([cell_remapped_path, weighted_cell_channel_path,
                             som_cluster_channel_avg_path, meta_cluster_channel_avg_path])

    # read in the remapping
    cell_remapped_data = pd.read_csv(cell_remapped_path)

    # assert the correct columns are contained
    misc_utils.verify_same_elements(
        remapped_data_cols=cell_remapped_data.columns.values,
        required_cols=['cell_som_cluster', 'cell_meta_cluster', 'cell_meta_cluster_rename']
    )

    # create the mapping from cell SOM to cell meta cluster
    # TODO: generating cell_remapped_dict and cell_renamed_meta_dict should be returned
    # to prevent repeat computation in summary file generation functions
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

    # re-compute the weighted channel average expression per cell meta cluster
    # add renamed meta cluster in
    print("Re-compute average weighted channel expression across cell meta clusters")
    cell_meta_cluster_channel_avg = compute_cell_cluster_weighted_channel_avg(
        fovs,
        channels,
        base_dir,
        weighted_cell_channel_name,
        cell_som_input_data,
        'cell_meta_cluster'
    )

    cell_meta_cluster_channel_avg['cell_meta_cluster_rename'] = \
        cell_meta_cluster_channel_avg['cell_meta_cluster'].map(cell_renamed_meta_dict)

    # re-save the weighted channel average expression per cell cluster
    cell_meta_cluster_channel_avg.to_csv(
        meta_cluster_channel_avg_path,
        index=False
    )

    # re-assign cell meta cluster labels back to the average weighted channel expression
    # per cell SOM cluster table
    print("Re-assigning meta cluster column in cell SOM cluster average weighted channel data")
    cell_som_cluster_channel_avg = pd.read_csv(som_cluster_channel_avg_path)

    cell_som_cluster_channel_avg['cell_meta_cluster'] = \
        cell_som_cluster_channel_avg['cell_som_cluster'].map(cell_remapped_dict)

    cell_som_cluster_channel_avg['cell_meta_cluster_rename'] = \
        cell_som_cluster_channel_avg['cell_meta_cluster'].map(cell_renamed_meta_dict)

    # re-save the cell SOM cluster average pixel cluster counts table
    cell_som_cluster_channel_avg.to_csv(som_cluster_channel_avg_path, index=False)


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
    io_utils.validate_paths([cell_cluster_channel_avg_path])

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
