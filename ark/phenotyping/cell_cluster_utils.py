import os
import subprocess

import feather
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from ark.analysis import visualize
from ark.utils import misc_utils, io_utils


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
    io_utils.validate_paths(os.path.join(base_dir, weighted_cell_channel_name))

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


def train_cell_som(fovs, channels, base_dir, pixel_data_dir, cell_table_path,
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
        pixel_data_dir (str):
            Name of directory with the pixel data with SOM and meta cluster labels added.
            Created by `pixel_consensus_cluster`.
        cell_table_path (str):
            Path of the cell table, needs to be created with `Segment_Image_Data.ipynb`
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
    pixel_data_path = os.path.join(base_dir, pixel_data_dir)
    cluster_counts_path = os.path.join(base_dir, cluster_counts_name)
    cluster_counts_norm_path = os.path.join(base_dir, cluster_counts_norm_name)
    weights_path = os.path.join(base_dir, weights_name)

    # check the cell table path and pixel data path exist
    io_utils.validate_paths([cell_table_path, pixel_data_path])

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
        fovs, pixel_data_path, cell_table_path, pixel_cluster_col
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

    if process.returncode != 0:
        raise OSError(
            "Process terminated: please view error messages displayed above for debugging."
        )

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
                  cell_data_name='cell_mat.feather',
                  pixel_cluster_col_prefix='pixel_meta_cluster_rename',
                  cell_som_cluster_count_avgs_name='cell_som_cluster_count_avgs.csv'):
    """Uses trained weights to assign cluster labels on full cell data.

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
        cell_data_name (str):
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
    cell_data_path = os.path.join(base_dir, cell_data_name)

    # check the path to the normalized pixel cluster counts per cell and weights file exists
    io_utils.validate_paths([cluster_counts_norm_path, weights_path])

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
                    weights_path, cell_data_path]

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

    if process.returncode != 0:
        raise OSError(
            "Process terminated: please view error messages displayed above for debugging."
        )

    # compute the average pixel SOM/meta counts per cell SOM cluster
    print("Computing the average number of pixel SOM/meta cluster counts per cell SOM cluster")
    cell_som_cluster_avgs_and_counts = compute_cell_cluster_count_avg(
        cell_data_path,
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
                           cell_data_name='cell_mat.feather',
                           cell_som_cluster_count_avgs_name='cell_som_cluster_avgs.csv',
                           cell_meta_cluster_count_avgs_name='cell_meta_cluster_avgs.csv',
                           weighted_cell_channel_name='weighted_cell_channel.csv',
                           cell_som_cluster_channel_avg_name='cell_som_cluster_channel_avg.csv',
                           cell_meta_cluster_channel_avg_name='cell_meta_cluster_channel_avg.csv',
                           clust_to_meta_name='cell_clust_to_meta.feather', seed=42):
    """Run consensus clustering algorithm on cell-level data averaged across each cell SOM cluster.

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
        cell_data_name (str):
            Name of the file containing the cell data with cell SOM cluster labels.
            Created by cluster_cells.
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
        seed (int):
            The random seed to set for consensus clustering
    """

    # define the paths to the data
    cell_data_path = os.path.join(base_dir, cell_data_name)
    som_cluster_counts_avg_path = os.path.join(base_dir, cell_som_cluster_count_avgs_name)
    weighted_channel_path = os.path.join(base_dir, weighted_cell_channel_name)
    clust_to_meta_path = os.path.join(base_dir, clust_to_meta_name)

    # check paths
    io_utils.validate_paths([cell_data_path, som_cluster_counts_avg_path, weighted_channel_path])

    # verify the pixel_cluster_col provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster_rename']
    )

    # run the consensus clustering process
    process_args = ['Rscript', '/cell_consensus_cluster.R', pixel_cluster_col,
                    str(max_k), str(cap), cell_data_path,
                    som_cluster_counts_avg_path, clust_to_meta_path, str(seed)]

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

    if process.returncode != 0:
        raise OSError(
            "Process terminated: please view error messages displayed above for debugging."
        )

    # compute the average pixel SOM/meta counts per cell meta cluster
    print("Compute the average number of pixel SOM/meta cluster counts per cell meta cluster")
    cell_meta_cluster_avgs_and_counts = compute_cell_cluster_count_avg(
        cell_data_path,
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
        cell_data_name,
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
        cell_data_name,
        'cell_meta_cluster'
    )

    # save the weighted channel average expression per cell cluster
    cell_meta_cluster_channel_avg.to_csv(
        os.path.join(base_dir, cell_meta_cluster_channel_avg_name),
        index=False
    )

    os.remove('Rplots.pdf')


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
    io_utils.validate_paths([cell_consensus_path, cell_remapped_path,
                             som_cluster_counts_avgs_path, meta_cluster_counts_avgs_path,
                             weighted_channel_path, som_cluster_channel_avgs_path,
                             meta_cluster_channel_avgs_path])

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
    io_utils.validate_paths(cell_cluster_channel_avg_path)

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


def add_consensus_labels_cell_table(base_dir, cell_table_path, cell_data_name):
    """Adds the consensus cluster labels to the cell table,
    then resaves data to `{cell_table_path}_cell_labels.csv`
    Args:
        base_dir (str):
            The path to the data directory
        cell_table_path (str):
            Path of the cell table, needs to be created with `Segment_Image_Data.ipynb`
        cell_data_name (str):
            Name of file with the cell consensus clustered results (both cell SOM and meta labels)
    """

    # define the data paths
    cell_data_path = os.path.join(base_dir, cell_data_name)

    # file path validation
    io_utils.validate_paths([cell_data_path, cell_data_path])

    # read in the data, ensure sorted by FOV column just in case
    cell_table = pd.read_csv(cell_table_path)
    consensus_data = feather.read_dataframe(cell_data_path)

    # for a simpler merge, rename segmentation_label to label in consensus_data
    consensus_data = consensus_data.rename(
        {'segmentation_label': 'label'}, axis=1
    )

    # merge the cell table with the consensus data to retrieve the meta clusters
    cell_table_merged = cell_table.merge(
        consensus_data, how='left', on=['fov', 'label']
    )

    # adjust column names and drop consensus data-specific columns
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
