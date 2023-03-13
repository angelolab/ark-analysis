import os

from alpineer import io_utils, misc_utils

from ark.phenotyping import cluster_helpers
from ark.phenotyping import cell_cluster_utils


def train_cell_som(fovs, base_dir, cell_table_path, cell_som_cluster_cols,
                   cell_som_input_data, som_weights_name='cell_som_weights.feather',
                   xdim=10, ydim=10, lr_start=0.05, lr_end=0.01, num_passes=1, seed=42,
                   overwrite=False):
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
        cell_som_input_data (pandas.DataFrame):
            The input data to use for SOM training
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
        seed (int):
            The random seed to use for training the SOM
        overwrite (bool):
            If set, force retrains the SOM and overwrites the weights

    Returns:
        cluster_helpers.CellSOMCluster:
            The SOM cluster object containing the cell SOM weights
    """

    # define the data paths
    som_weights_path = os.path.join(base_dir, som_weights_name)

    # check the cell table path exists
    io_utils.validate_paths([cell_table_path])

    # verify the cell_som_cluster_cols columns provided are valid
    misc_utils.verify_in_list(
        provided_cluster_cols=cell_som_cluster_cols,
        som_input_cluster_cols=cell_som_input_data.columns.values
    )

    # define the cell SOM cluster object
    cell_pysom = cluster_helpers.CellSOMCluster(
        cell_som_input_data, som_weights_path, fovs, cell_som_cluster_cols,
        num_passes=num_passes, xdim=xdim, ydim=ydim, lr_start=lr_start, lr_end=lr_end,
        seed=seed
    )

    # train the SOM weights
    # NOTE: seed has to be set in cyFlowSOM.pyx, done by passing flag in PixieSOMCluster
    print("Training SOM")
    cell_pysom.train_som(overwrite=overwrite)

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

    Returns:
        pandas.DataFrame:
            The cell data in `cell_pysom.cell_data` with SOM labels assigned
    """

    # raise error if weights haven't been assigned to cell_pysom
    if cell_pysom.weights is None:
        raise ValueError("Using untrained cell_pysom object, please invoke train_cell_som first")

    # non-pixel cluster inputs won't be cell size normalized
    cols_to_drop = ['fov', 'segmentation_label']
    if 'cell_size' in cell_pysom.cell_data.columns.values:
        cols_to_drop.append('cell_size')

    # ensure the weights columns are valid indexes, do so by ensuring
    # the cluster_counts_norm and weights columns are the same
    # minus the metadata columns (and possibly cluster col) that appear in cluster_counts_norm
    if 'cell_som_cluster' in cell_pysom.cell_data.columns.values:
        cols_to_drop.append('cell_som_cluster')

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

    return cell_data_som_labels


def generate_som_avg_files(base_dir, cell_som_input_data, cell_som_cluster_cols,
                           cell_som_expr_col_avg_name, overwrite=False):
    """Computes and saves the average expression of all `cell_som_cluster_cols`
    across cell SOM clusters.

    Args:
        base_dir (str):
            The path to the data directory
        cell_som_input_data (pandas.DataFrame):
            The input data used for SOM training with SOM labels attached
        cell_som_cluster_cols (list):
            The list of columns used for SOM training
        cell_som_expr_col_avg_name (str):
            The name of the file to write the average expression per column
            across cell SOM clusters
        overwrite (bool):
            If set, regenerate the averages of `cell_som_cluster_columns` for SOM clusters
    """

    # define the paths to the data
    som_expr_col_avg_path = os.path.join(base_dir, cell_som_expr_col_avg_name)

    # raise error if cell_som_input_data doesn't contain SOM labels
    if 'cell_som_cluster' not in cell_som_input_data.columns.values:
        raise ValueError('cell_som_input_data does not have SOM labels assigned')

    # if the channel SOM average file already exists and the overwrite flag isn't set, skip
    if os.path.exists(som_expr_col_avg_path):
        if not overwrite:
            print("Already generated average expression file for each cell SOM column, skipping")
            return

        print(
            "Overwrite flag set, regenerating average expression file for cell SOM clusters"
        )

    # compute the average column expression values per cell SOM cluster
    print("Computing the average value of each training column specified per cell SOM cluster")
    cell_som_cluster_avgs = cell_cluster_utils.compute_cell_som_cluster_cols_avg(
        cell_som_input_data,
        cell_som_cluster_cols,
        'cell_som_cluster',
        keep_count=True
    )

    # save the average expression values of cell_som_cluster_cols per cell SOM cluster
    cell_som_cluster_avgs.to_csv(
        som_expr_col_avg_path,
        index=False
    )