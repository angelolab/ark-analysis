import os
import warnings

import feather
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from tmi import io_utils, misc_utils

from ark.analysis import visualize
from ark.phenotyping import cluster_helpers


def compute_cell_cluster_expr_avg(cell_cluster_path, cell_som_cluster_cols,
                                  cell_cluster_col, keep_count=False):
    """For each cell SOM cluster, compute the average expression of all `cell_som_cluster_cols`

    Args:
        cell_cluster_path (str):
            The path to the cell data with SOM and/or meta labels, created by `cluster_cells`
        cell_som_cluster_cols (list):
            The list of columns used for SOM training
        cell_cluster_col (str):
            Name of the cell cluster column to group by,
            should be `'cell_som_cluster'` or `'cell_meta_cluster'`
        keep_count (bool):
            Whether to include the cell counts or not,
            should only be set to `True` for visualization support

    Returns:
        pandas.DataFrame:
            Contains the average values for each column across cell SOM clusters
    """

    # Validate paths
    io_utils.validate_paths(cell_cluster_path)

    # verify the cell cluster col prefix specified is valid
    misc_utils.verify_in_list(
        provided_cluster_col=cell_cluster_col,
        valid_cluster_cols=['cell_som_cluster', 'cell_meta_cluster']
    )

    # read in the clustered data
    cluster_data = feather.read_dataframe(cell_cluster_path)

    # verify that the cluster columns are valid
    misc_utils.verify_in_list(
        provided_cluster_col=cell_som_cluster_cols,
        cluster_data_valid_cols=cluster_data.columns.values
    )

    # subset the data by columns used for SOM training, as well as the cell SOM assignments
    cluster_data_subset = cluster_data.loc[:, list(cell_som_cluster_cols) + [cell_cluster_col]]

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
                                     cell_cluster_name='cell_som_input_data.feather',
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
            The name of the weighted cell table, created in `3_Pixie_Cluster_Cells.ipynb`
        cell_cluster_name (str):
            Name of the file containing the cell data with cluster labels
        cell_cluster_col (str):
            Whether to aggregate by cell SOM or meta labels
            Needs to be either 'cell_som_cluster', or 'cell_meta_cluster'

    Returns:
        pandas.DataFrame:
            Each cell cluster mapped to the average expression for each marker
    """

    weighted_cell_channel_name_path: str = os.path.join(base_dir, weighted_cell_channel_name)
    cell_cluster_name_path: str = os.path.join(base_dir, cell_cluster_name)

    # verify the cell table actually exists
    io_utils.validate_paths([weighted_cell_channel_name_path, cell_cluster_name_path])

    # verify the cell cluster col specified is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[cell_cluster_col],
        valid_cluster_cols=['cell_som_cluster', 'cell_meta_cluster']
    )

    # read the weighted cell channel table in
    cell_table = feather.read_dataframe(weighted_cell_channel_name_path)

    # subset on only the fovs the user has specified
    cell_table = cell_table[cell_table['fov'].isin(fovs)]

    # read the clustered data
    cluster_data = feather.read_dataframe(cell_cluster_name_path)

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


def create_c2pc_data(fovs, pixel_data_path, cell_table_path,
                     pixel_cluster_col='pixel_meta_cluster_rename'):
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


def train_cell_som(fovs, base_dir, cell_table_path, cell_som_cluster_cols,
                   cell_som_input_data_name, som_weights_name='cell_som_weights.feather',
                   xdim=10, ydim=10, lr_start=0.05, lr_end=0.01, num_passes=1):
    """Run the SOM training on the expression columns specified in `cell_som_cluster_cols`.

    Saves the SOM weights to `base_dir/som_weights_name`.

    Args:
        fovs (list):
            The list of fovs to subset on
        base_dir (str):
            The path to the data directories
        cell_table_path (str):
            Path of the cell table, needs to be created with `Segment_Image_Data.ipynb`
        cell_som_cluster_cols (list):
            The list of columns in `cell_som_input_data_name` to use for SOM training
        cell_som_input_data_name (str):
            The input file to use for SOM training
        som_weights_name (str):
            The name of the file to save the SOM weights to
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

    Returns:
        cluster_helpers.CellSOMCluster:
            The SOM cluster object containing the cell SOM weights
    """

    # define the data paths
    cell_som_input_data_path = os.path.join(base_dir, cell_som_input_data_name)
    som_weights_path = os.path.join(base_dir, som_weights_name)

    # check the cell SOM inputs cell table path exists
    io_utils.validate_paths([cell_som_input_data_path, cell_table_path])

    # load the cell SOM input data for verification
    cell_som_input_data = feather.read_dataframe(cell_som_input_data_path)

    # verify the cell_som_cluster_cols columns provided are valid
    misc_utils.verify_in_list(
        provided_cluster_cols=cell_som_cluster_cols,
        som_input_cluster_cols=cell_som_input_data.columns.values
    )

    # define the cell SOM cluster object
    cell_pysom = cluster_helpers.CellSOMCluster(
        cell_som_input_data_path, som_weights_path, cell_som_cluster_cols,
        num_passes=num_passes, xdim=xdim, ydim=ydim, lr_start=lr_start, lr_end=lr_end
    )

    # train the SOM weights
    # NOTE: seed has to be set in cyFlowSOM.pyx, done by passing flag in PixieSOMCluster
    print("Training SOM")
    cell_pysom.train_som()

    return cell_pysom


def cluster_cells(base_dir, cell_pysom, cell_som_cluster_cols):
    """Uses trained SOM weights to assign cluster labels on full cell data.

    Saves data with cluster labels to `cell_cluster_name`.

    Args:
        base_dir (str):
            The path to the data directory
        cell_pysom (cluster_helpers.CellSOMCluster):
            The SOM cluster object containing the cell SOM weights
        cell_som_cluster_cols (list):
            The list of columns used for SOM training
    """

    # raise error if weights haven't been assigned to cell_pysom
    if cell_pysom.weights is None:
        raise ValueError("Using untrained cell_pysom object, please invoke train_cell_som first")

    # non-pixel cluster inputs won't be cell size normalized
    cols_to_drop = ['fov', 'segmentation_label']
    if 'cell_size' in cell_pysom.cell_data.columns.values:
        cols_to_drop.append('cell_size')

    # ensure the weights columns are valid indexes, do so by ensuring
    # the cell_som_input_data and weights columns are the same
    # minus the metadata columns that appear in cluster_counts_norm
    cell_som_input_data = cell_pysom.cell_data.drop(
        columns=cols_to_drop
    )

    # handles the case if user specifies a subset of columns for generic cell clustering
    # NOTE: CellSOMCluster ensures column ordering by using the preset self.columns as an index
    misc_utils.verify_in_list(
        cell_weights_columns=cell_pysom.weights.columns.values,
        cell_som_input_data_columns=cell_som_input_data.columns.values
    )

    # run the trained SOM on the dataset, assigning clusters
    print("Mapping cell data to SOM cluster labels")
    cell_data_som_labels = cell_pysom.assign_som_clusters()

    # resave cell_data
    os.remove(cell_pysom.cell_data_path)
    feather.write_dataframe(cell_data_som_labels, cell_pysom.cell_data_path)


def generate_som_avg_files(base_dir, cell_pysom, cell_som_cluster_cols,
                           cell_som_expr_col_avg_name):
    """Computes and saves the average expression of all `cell_som_cluster_cols`
    across cell SOM clusters.

    Args:
        base_dir (str):
            The path to the data directory
        cell_pysom (cluster_helpers.PixelSOMCluster):
            The SOM cluster object containing the pixel SOM weights
        cell_som_cluster_cols (list):
            The list of columns used for SOM training
        cell_som_expr_col_avg_name (str):
            The name of the file to write the average expression per column
            across cell SOM clusters
    """

    # define the paths to the data
    som_expr_col_avg_path = os.path.join(base_dir, cell_som_expr_col_avg_name)

    # raise error if weights haven't been assigned to pixel_pysom
    if cell_pysom.weights is None:
        raise ValueError("Using untrained cell_pysom object, please invoke train_som first")

    # if the channel SOM average file already exists, skip
    if os.path.exists(som_expr_col_avg_path):
        print("Already generated average expression file for each cell SOM column, skipping")
        return

    # compute the average column expression values per cell SOM cluster
    print("Computing the average value of each training column specified per cell SOM cluster")
    cell_som_cluster_avgs = compute_cell_cluster_expr_avg(
        cell_pysom.cell_data_path,
        cell_som_cluster_cols,
        'cell_som_cluster',
        keep_count=True
    )

    # save the average pixel SOM/meta counts per cell SOM cluster
    cell_som_cluster_avgs.to_csv(
        som_expr_col_avg_path,
        index=False
    )


def cell_consensus_cluster(base_dir, cell_som_cluster_cols, cell_som_input_data_name,
                           cell_som_expr_col_avg_name, max_k=20, cap=3, seed=42):
    """Run consensus clustering algorithm on cell-level data averaged across each cell SOM cluster.

    Saves data with consensus cluster labels to cell_consensus_name.

    Args:
        base_dir (str):
            The path to the data directory
        cell_som_cluster_cols (list):
            The list of columns used for SOM training
        cell_som_input_data_name (str):
            The input file used for SOM training
        cell_som_expr_col_avg_name (str):
            The name of the file with the average expression per column across cell SOM clusters.
            Used to run consensus clustering on.
        max_k (int):
            The number of consensus clusters
        cap (int):
            z-score cap to use when hierarchical clustering
        seed (int):
            The random seed to set for consensus clustering

    Returns:
        cluster_helpers.PixieConsensusCluster:
            The consensus cluster object containing the SOM to meta mapping
    """
    # define the paths to the data
    cell_som_input_data_path = os.path.join(base_dir, cell_som_input_data_name)
    som_expr_col_avg_path = os.path.join(base_dir, cell_som_expr_col_avg_name)

    # check paths
    io_utils.validate_paths([cell_som_input_data_path, som_expr_col_avg_path])

    # load in the cell SOM average expression data
    cluster_count_sub = pd.read_csv(som_expr_col_avg_path, nrows=1)

    # verify the SOM cluster cols provided exist in cluster_count_sub
    misc_utils.verify_in_list(
        provided_cluster_cols=cell_som_cluster_cols,
        som_cluster_counts_cols=cluster_count_sub.columns.values
    )

    # define the cell consensus cluster object
    cell_cc = cluster_helpers.PixieConsensusCluster(
        'cell', som_expr_col_avg_path, cell_som_cluster_cols, max_k=max_k, cap=cap
    )

    # z-score and cap the data
    print("z-score scaling and capping data")
    cell_cc.scale_data()

    # set random seed for consensus clustering
    np.random.seed(seed)

    # run consensus clustering
    print("Running consensus clustering")
    cell_cc.run_consensus_clustering()

    # generate the som to meta cluster map
    print("Mapping cell data to consensus cluster labels")
    cell_cc.generate_som_to_meta_map()

    # assign the consensus cluster labels to cell_som_input_data_path data and resave
    cell_data = feather.read_dataframe(cell_som_input_data_path)
    cell_meta_assign = cell_cc.assign_consensus_labels(cell_data)
    feather.write_dataframe(
        cell_meta_assign,
        cell_som_input_data_path,
        compression='uncompressed'
    )

    return cell_cc


def generate_meta_avg_files(base_dir, cell_cc, cell_som_cluster_cols,
                            cell_som_input_data_name,
                            cell_som_expr_col_avg_name,
                            cell_meta_expr_col_avg_name):
    """Computes and saves the average cluster column expression across pixel meta clusters.
    Assigns meta cluster labels to the data stored in `cell_som_expr_col_avg_name`.

    Args:
        base_dir (str):
            The path to the data directory
        cell_cc (cluster_helpers.PixieConsensusCluster):
            The consensus cluster object containing the SOM to meta mapping
        cell_som_cluster_cols (list):
            The list of columns used for SOM training
        cell_som_input_data_name (str):
            The input file used for SOM training.
            Will have meta labels appended after this process is run.
        cell_som_expr_col_avg_name (str):
            The average values of `cell_som_cluster_cols` per cell SOM cluster.
            Used to run consensus clustering on.
        cell_meta_expr_col_avg_name (str):
            Same as above except for cell meta clusters
    """
    # define the paths to the data
    cell_som_input_data_path = os.path.join(base_dir, cell_som_input_data_name)
    som_expr_col_avg_path = os.path.join(base_dir, cell_som_expr_col_avg_name)
    meta_expr_col_avg_path = os.path.join(base_dir, cell_meta_expr_col_avg_name)

    # check paths
    io_utils.validate_paths([cell_som_input_data_path, som_expr_col_avg_path])

    # if the column average file for cell meta clusters already exists, skip
    if os.path.exists(meta_expr_col_avg_path):
        print("Already generated average expression file for cell meta clusters, skipping")
        return

    # compute the average pixel SOM/meta counts per cell meta cluster
    print("Computing the average value of each training column specified per cell meta cluster")
    cell_meta_cluster_avgs = compute_cell_cluster_expr_avg(
        cell_som_input_data_path,
        cell_som_cluster_cols,
        'cell_meta_cluster',
        keep_count=True
    )

    # save the average pixel SOM/meta counts per cell meta cluster
    cell_meta_cluster_avgs.to_csv(
        meta_expr_col_avg_path,
        index=False
    )

    print(
        "Mapping meta cluster values onto average expression values across cell SOM clusters"
    )

    # read in the average number of pixel/SOM clusters across all cell SOM clusters
    cell_som_cluster_avgs = pd.read_csv(som_expr_col_avg_path)

    # merge metacluster assignments in
    cell_som_cluster_avgs = pd.merge_asof(
        cell_som_cluster_avgs, cell_cc.mapping, on='cell_som_cluster'
    )

    # resave average number of pixel/SOM clusters across all cell SOM clusters
    # with metacluster assignments
    cell_som_cluster_avgs.to_csv(
        som_expr_col_avg_path,
        index=False
    )


def generate_wc_avg_files(fovs, channels, base_dir, cell_cc,
                          cell_som_input_data_name='cell_som_input_data.feather',
                          weighted_cell_channel_name='weighted_cell_channel.feather',
                          cell_som_cluster_channel_avg_name='cell_som_cluster_channel_avg.csv',
                          cell_meta_cluster_channel_avg_name='cell_meta_cluster_channel_avg.csv'):
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
        cell_som_input_data_name (str):
            The input file used for SOM training. For weighted channel averaging, it should
            contain the number of pixel SOM/meta cluster counts of each cell,
            normalized by `cell_size`
        weighted_cell_channel_name (str):
            The name of the file containing the weighted channel expression table
        cell_som_cluster_channel_avg_name (str):
            The name of the file to save the average weighted channel expression
            per cell SOM cluster
        cell_meta_cluster_channel_avg_name (str):
            Same as above except for cell meta clusters
    """
    # define the paths to the data
    cell_som_input_data_path = os.path.join(base_dir, cell_som_input_data_name)
    weighted_channel_path = os.path.join(base_dir, weighted_cell_channel_name)
    som_cluster_channel_avg_path = os.path.join(base_dir, cell_som_cluster_channel_avg_name)
    meta_cluster_channel_avg_path = os.path.join(base_dir, cell_meta_cluster_channel_avg_name)

    # check paths
    io_utils.validate_paths([cell_som_input_data_path, weighted_channel_path])

    # if the weighted channel average files exist, skip
    if os.path.exists(som_cluster_channel_avg_path) and \
       os.path.exists(meta_cluster_channel_avg_path):
        print("Already generated average weighted channel expression files, skipping")
        return

    print("Compute average weighted channel expression across cell SOM clusters")
    cell_som_cluster_channel_avg = compute_cell_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        weighted_cell_channel_name,
        cell_som_input_data_name,
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
    cell_meta_cluster_channel_avg = compute_cell_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        weighted_cell_channel_name,
        cell_som_input_data_name,
        'cell_meta_cluster'
    )

    # save the weighted channel average expression per cell cluster
    cell_meta_cluster_channel_avg.to_csv(
        meta_cluster_channel_avg_path,
        index=False
    )


def apply_cell_meta_cluster_remapping(base_dir, cell_som_input_data_name, cell_remapped_name):
    """Apply the meta cluster remapping to the data in `cell_consensus_name`.
    Resave the re-mapped consensus data to `cell_consensus_name`.

    Args:
        base_dir (str):
            The path to the data directory
        cell_som_input_data_name (str):
            The input file used for SOM training
        cell_remapped_name (str):
            Name of the file containing the cell SOM clusters to their remapped meta clusters
    """

    # define the data paths
    cell_som_input_data_path = os.path.join(base_dir, cell_som_input_data_name)
    cell_remapped_path = os.path.join(base_dir, cell_remapped_name)

    # file path validation
    io_utils.validate_paths([cell_som_input_data_path, cell_remapped_path])

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

    # load the cell consensus data in
    print("Using re-mapping scheme to re-label cell meta clusters")
    cell_consensus_data = feather.read_dataframe(cell_som_input_data_path)

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
    feather.write_dataframe(
        cell_consensus_data,
        cell_som_input_data_path,
        compression='uncompressed'
    )


def generate_remap_avg_count_files(base_dir, cell_som_input_data_name,
                                   cell_remapped_name, cell_som_cluster_cols,
                                   cell_som_expr_col_avg_name,
                                   cell_meta_expr_col_avg_name):
    """Apply the cell cluster remapping to the average count files

    Args:
        base_dir (str):
            The path to the data directory
        cell_som_input_data_name (str):
            The input file used for SOM training
        cell_remapped_name (str):
            Name of the file containing the cell SOM clusters to their remapped meta clusters
        cell_som_cluster_cols (list):
            The list of columns used for SOM training
        cell_som_expr_col_avg_name (str):
            The average values of `cell_som_cluster_cols` per cell SOM cluster
        cell_meta_expr_col_avg_name (str):
            Same as above except for cell meta clusters
    """
    # define the data paths
    cell_som_input_data_path = os.path.join(base_dir, cell_som_input_data_name)
    cell_remapped_path = os.path.join(base_dir, cell_remapped_name)
    som_expr_col_avg_path = os.path.join(base_dir, cell_som_expr_col_avg_name)
    meta_expr_col_avg_path = os.path.join(base_dir, cell_meta_expr_col_avg_name)

    # file path validation
    io_utils.validate_paths([cell_som_input_data_path, cell_remapped_path,
                             som_expr_col_avg_path, meta_expr_col_avg_path])

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

    # re-compute the average number of pixel SOM/meta clusters per cell meta cluster
    # add renamed meta cluster in
    print("Re-compute pixel SOM/meta cluster count per cell meta cluster")
    cell_meta_cluster_avgs = compute_cell_cluster_expr_avg(
        cell_som_input_data_path,
        cell_som_cluster_cols,
        'cell_meta_cluster',
        keep_count=True
    )

    cell_meta_cluster_avgs['cell_meta_cluster_rename'] = \
        cell_meta_cluster_avgs['cell_meta_cluster'].map(cell_renamed_meta_dict)

    # re-save the average number of pixel SOM/meta clusters per cell meta cluster
    cell_meta_cluster_avgs.to_csv(
        meta_expr_col_avg_path,
        index=False
    )

    # re-assign cell meta cluster labels back to the average pixel cluster counts
    # per cell SOM cluster table
    print("Re-assigning meta cluster column in cell SOM cluster average pixel cluster counts data")
    cell_som_cluster_avgs = pd.read_csv(som_expr_col_avg_path)

    cell_som_cluster_avgs['cell_meta_cluster'] = \
        cell_som_cluster_avgs['cell_som_cluster'].map(cell_remapped_dict)

    cell_som_cluster_avgs['cell_meta_cluster_rename'] = \
        cell_som_cluster_avgs['cell_meta_cluster'].map(cell_renamed_meta_dict)

    # re-save the cell SOM cluster average pixel cluster counts table
    cell_som_cluster_avgs.to_csv(som_expr_col_avg_path, index=False)


def generate_remap_avg_wc_files(fovs, channels, base_dir, cell_som_input_data_name,
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
        cell_som_input_data_name (str):
            The input file used for SOM training. For weighted channel averaging, this should
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
    cell_som_input_data_path = os.path.join(base_dir, cell_som_input_data_name)
    cell_remapped_path = os.path.join(base_dir, cell_remapped_name)
    weighted_cell_channel_path = os.path.join(base_dir, weighted_cell_channel_name)
    som_cluster_channel_avg_path = os.path.join(base_dir, cell_som_cluster_channel_avg_name)
    meta_cluster_channel_avg_path = os.path.join(base_dir, cell_meta_cluster_channel_avg_name)

    # file path validation
    io_utils.validate_paths([cell_som_input_data_path, cell_remapped_path,
                             weighted_cell_channel_path, som_cluster_channel_avg_path,
                             meta_cluster_channel_avg_path])

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
    cell_meta_cluster_channel_avg = compute_cell_cluster_channel_avg(
        fovs,
        channels,
        base_dir,
        weighted_cell_channel_name,
        cell_som_input_data_name,
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


def add_consensus_labels_cell_table(base_dir, cell_table_path, cell_som_input_data_name):
    """Adds the consensus cluster labels to the cell table,
    then resaves data to `{cell_table_path}_cell_labels.csv`

    Args:
        base_dir (str):
            The path to the data directory
        cell_table_path (str):
            Path of the cell table, needs to be created with `Segment_Image_Data.ipynb`
        cell_som_input_data_name (str):
            The input file used for SOM training
    """

    # define the data paths
    cell_som_input_data_path = os.path.join(base_dir, cell_som_input_data_name)

    # file path validation
    io_utils.validate_paths([cell_table_path, cell_som_input_data_path])

    # read in the data, ensure sorted by FOV column just in case
    cell_table = pd.read_csv(cell_table_path)
    consensus_data = feather.read_dataframe(cell_som_input_data_path)

    # for a simpler merge, rename segmentation_label to label in consensus_data
    consensus_data = consensus_data.rename(
        {'segmentation_label': 'label'}, axis=1
    )

    # merge the cell table with the consensus data to retrieve the meta clusters
    cell_table_merged = cell_table.merge(
        consensus_data, how='left', on=['fov', 'label']
    )

    # adjust column names and drop consensus data-specific columns
    # NOTE: non-pixel cluster inputs will not have the cell size attribute for normalization
    if 'cell_size_y' in cell_table_merged.columns.values:
        cell_table_merged = cell_table_merged.drop(columns=['cell_size_y'])
        cell_table_merged = cell_table_merged.rename(
            {'cell_size_x': 'cell_size'}, axis=1
        )

    # subset on just the cell table columns plus the meta cluster rename column
    # NOTE: rename cell_meta_cluster_rename to just cell_meta_cluster for simplicity
    cell_table_merged = cell_table_merged[
        list(cell_table.columns.values) + ['cell_meta_cluster_rename']
    ]
    cell_table_merged = cell_table_merged.rename(
        {'cell_meta_cluster_rename': 'cell_meta_cluster'}, axis=1
    )

    # fill any N/A cell_meta_cluster values with 'Unassigned'
    # NOTE: this happens when a cell is so small no pixel clusters are detected inside of them
    cell_table_merged['cell_meta_cluster'] = cell_table_merged['cell_meta_cluster'].fillna(
        'Unassigned'
    )

    # resave cell table with new meta cluster column
    new_cell_table_path = os.path.splitext(cell_table_path)[0] + '_cell_labels.csv'
    cell_table_merged.to_csv(new_cell_table_path, index=False)
