import os

import numpy as np
import pandas as pd
from alpineer import io_utils, misc_utils

from ark.phenotyping import cluster_helpers
from ark.phenotyping import cell_cluster_utils


def cell_consensus_cluster(base_dir, cell_som_cluster_cols, cell_som_input_data,
                           cell_som_expr_col_avg_name, max_k=20, cap=3, seed=42):
    """Run consensus clustering algorithm on cell-level data averaged across each cell SOM cluster.

    Saves data with consensus cluster labels to cell_consensus_name.

    Args:
        base_dir (str):
            The path to the data directory
        cell_som_cluster_cols (list):
            The list of columns used for SOM training
        cell_som_input_data (pandas.DataFrame):
            The data used for SOM training with SOM labels attached
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
        tuple:
            - cluster_helpers.PixieConsensusCluster: the consensus cluster object containing the
              SOM to meta mapping
            - pandas.DataFrame: the input data used for SOM training with meta labels attached
    """
    # define the paths to the data
    som_expr_col_avg_path = os.path.join(base_dir, cell_som_expr_col_avg_name)

    # check paths
    io_utils.validate_paths([som_expr_col_avg_path])

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

    # assign the consensus cluster labels to cell_som_input_data
    cell_meta_assign = cell_cc.assign_consensus_labels(cell_som_input_data)

    return cell_cc, cell_meta_assign


def generate_meta_avg_files(base_dir, cell_cc, cell_som_cluster_cols,
                            cell_som_input_data,
                            cell_som_expr_col_avg_name,
                            cell_meta_expr_col_avg_name, overwrite=False):
    """Computes and saves the average cluster column expression across pixel meta clusters.
    Assigns meta cluster labels to the data stored in `cell_som_expr_col_avg_name`.

    Args:
        base_dir (str):
            The path to the data directory
        cell_cc (cluster_helpers.PixieConsensusCluster):
            The consensus cluster object containing the SOM to meta mapping
        cell_som_cluster_cols (list):
            The list of columns used for SOM training
        cell_som_input_data (pandas.DataFrame):
            The input data used for SOM training.
            Will have meta labels appended after this process is run.
        cell_som_expr_col_avg_name (str):
            The average values of `cell_som_cluster_cols` per cell SOM cluster.
            Used to run consensus clustering on.
        cell_meta_expr_col_avg_name (str):
            Same as above except for cell meta clusters
        overwrite (bool):
            If set, regenerate the averages of `cell_som_cluster_cols` per meta cluster
    """
    # define the paths to the data
    som_expr_col_avg_path = os.path.join(base_dir, cell_som_expr_col_avg_name)
    meta_expr_col_avg_path = os.path.join(base_dir, cell_meta_expr_col_avg_name)

    # check paths
    io_utils.validate_paths([som_expr_col_avg_path])

    # raise error if cell_som_input_data doesn't contain meta labels
    if 'cell_meta_cluster' not in cell_som_input_data.columns.values:
        raise ValueError('cell_som_input_data does not have meta labels assigned')

    # if the column average file for cell meta clusters already exists, skip
    if os.path.exists(meta_expr_col_avg_path):
        if not overwrite:
            print("Already generated average expression file for cell meta clusters, skipping")
            return

        print(
            "Overwrite flag set, regenerating average expression file for cell meta clusters"
        )

    # compute the average value of each expression column per cell meta cluster
    print("Computing the average value of each training column specified per cell meta cluster")
    cell_meta_cluster_avgs = cell_cluster_utils.compute_cell_som_cluster_cols_avg(
        cell_som_input_data,
        cell_som_cluster_cols,
        'cell_meta_cluster',
        keep_count=True
    )

    # save the average expression values of cell_som_cluster_cols per cell meta cluster
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


def apply_cell_meta_cluster_remapping(base_dir, cell_som_input_data, cell_remapped_name):
    """Apply the meta cluster remapping to the data in `cell_consensus_name`.
    Resave the re-mapped consensus data to `cell_consensus_name`.

    Args:
        base_dir (str):
            The path to the data directory
        cell_som_input_data (pandas.DataFrame):
            The input data used for SOM training
        cell_remapped_name (str):
            Name of the file containing the cell SOM clusters to their remapped meta clusters

    Returns:
        pandas.DataFrame:
            The input data used for SOM training with renamed meta labels attached
    """

    # define the data paths
    cell_remapped_path = os.path.join(base_dir, cell_remapped_name)

    # file path validation
    io_utils.validate_paths([cell_remapped_path])

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

    # load the cell consensus data in
    print("Using re-mapping scheme to re-label cell meta clusters")
    # ensure that no SOM clusters are missing from the mapping
    misc_utils.verify_in_list(
        fov_som_labels=cell_som_input_data['cell_som_cluster'],
        som_labels_in_mapping=list(cell_remapped_dict.keys())
    )

    # assign the new meta cluster labels
    cell_som_input_data['cell_meta_cluster'] = \
        cell_som_input_data['cell_som_cluster'].map(cell_remapped_dict)

    # assign the new renamed meta cluster names
    # assign the new meta cluster labels
    cell_som_input_data['cell_meta_cluster_rename'] = \
        cell_som_input_data['cell_meta_cluster'].map(cell_renamed_meta_dict)

    return cell_som_input_data


def generate_remap_avg_count_files(base_dir, cell_som_input_data,
                                   cell_remapped_name, cell_som_cluster_cols,
                                   cell_som_expr_col_avg_name,
                                   cell_meta_expr_col_avg_name):
    """Apply the cell cluster remapping to the average count files

    Args:
        base_dir (str):
            The path to the data directory
        cell_som_input_data (pandas.DataFrame):
            The input data used for SOM training
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
    cell_remapped_path = os.path.join(base_dir, cell_remapped_name)
    som_expr_col_avg_path = os.path.join(base_dir, cell_som_expr_col_avg_name)
    meta_expr_col_avg_path = os.path.join(base_dir, cell_meta_expr_col_avg_name)

    # file path validation
    io_utils.validate_paths([cell_remapped_path, som_expr_col_avg_path, meta_expr_col_avg_path])

    # read in the remapping
    cell_remapped_data = pd.read_csv(cell_remapped_path)

    # assert the correct columns are contained
    misc_utils.verify_same_elements(
        remapped_data_cols=cell_remapped_data.columns.values,
        required_cols=['cell_som_cluster', 'cell_meta_cluster', 'cell_meta_cluster_rename']
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

    # re-compute the average value of each expression column per meta cluster
    # add renamed meta cluster in
    print("Re-compute average value of each training column specified per cell meta cluster")
    cell_meta_cluster_avgs = cell_cluster_utils.compute_cell_som_cluster_cols_avg(
        cell_som_input_data,
        cell_som_cluster_cols,
        'cell_meta_cluster',
        keep_count=True
    )

    cell_meta_cluster_avgs['cell_meta_cluster_rename'] = \
        cell_meta_cluster_avgs['cell_meta_cluster'].map(cell_renamed_meta_dict)

    # re-save the average expression value of all cell SOM columns specified per cell meta cluster
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
