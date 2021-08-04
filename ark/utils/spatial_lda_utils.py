import pandas as pd

from ark.settings import BASE_COLS
from ark.utils.misc_utils import verify_in_list


def check_format_cell_table_args(cell_table, markers, clusters):
    """Checks the input arguments of the format_cell_table() function.

    Args:
        cell_table (pandas.DataFrame):
            A DataFrame containing the columns of cell marker frequencies and/or cluster ids.
        markers (list):
            A list of strings corresponding to marker names.
        clusters (list):
            A list of integers corresponding to cluster ids.
    """

    # Check cell table columns
    verify_in_list(cell_table_columns=cell_table.columns.to_list(), required_columns=BASE_COLS)

    # Check markers/clusters
    if markers is None and clusters is None:
        raise ValueError("markers and clusters cannot both be None")
    if markers is not None:
        if len(markers) == 0:
            raise ValueError("list of marker names cannot be empty")
        verify_in_list(markers=markers, cell_table_columns=cell_table.columns.to_list())
    if clusters is not None:
        if len(clusters) == 0:
            raise ValueError("list of cluster ids cannot be empty")
        if not isinstance(clusters, list) or not all(
                [isinstance(x, int) for x in clusters]):
            raise TypeError("clusters must be a list of integers")


def check_featurize_cell_table_args(cell_table, featurization, radius, cell_index):
    """Checks the input arguments of the featurize_cell_table() function.

    Args:
        cell_table (dict):
            A dictionary whose elements are the correctly formatted dataframes for each field of
            view.
        featurization (str):
            One of "cluster", "marker", "avg_marker", or "count".
        radius (int):
            Pixel radius corresponding to cellular neighborhood size.
        cell_index (str):
            Name of the column in each field of view pd.Dataframe indicating reference cells.
    """
    # Check valid data types
    if not isinstance(cell_table, dict):
        raise TypeError("cell_table should be of type 'dict'")
    if not isinstance(cell_table[0], pd.DataFrame):
        raise TypeError("cell_table should contain formatted dataframes")
    if not isinstance(radius, int):
        raise TypeError("radius should be of type 'int'")

    verify_in_list(featurization=[featurization],
                   featurization_options=["cluster", "marker", "avg_marker", "count"])
    verify_in_list(cell_index=[cell_index], cell_table_columns=cell_table[0].columns.to_list())

    if radius < 25:
        raise ValueError("radius must not be less than 25")


def check_create_difference_matrices_args(cell_table, features, training, inference):
    """Checks the input arguments of the create_difference_matrices() function.

    Args:
        cell_table (dict):
            A dictionary whose elements are the correctly formatted DataFrames for each field of
            view.
        features (dict):
            A dictionary containing the featurized cell table and the training data.
            Specifically, this is the output from
            :func:`~ark.spLDA.processing.featurize_cell_table`.
        training (bool):
            If True, create the difference matrix for running training algorithm.
        inference (bool):
             If True, create the difference matrix for running inference algorithm.
    """

    if not isinstance(cell_table, dict):
        raise TypeError("cell_table must be of type 'dict'")
    if not isinstance(features, dict):
        raise TypeError("features must be of type 'dict'")
    if not training and not inference:
        raise ValueError("One or both of 'training' or 'inference' must be True")
    if training and features["train_features"] is None:
        raise ValueError("train_features cannot be 'None'")
