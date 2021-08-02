import pandas as pd

from ark.settings import BASE_COLS


def check_format_cell_table_args(cell_table, markers, clusters):
    """Checks the input arguments of the format_cell_table() function.

    Args:
        cell_table (pd.DataFrame):
            A pandas DataFrame containing the columns of cell marker
            frequencies and/or cluster ids.
        markers (list, str):
            A list of strings corresponding to marker names.
        clusters (list, int):
            A list of integers corresponding to cluster ids.

    """

    # Check cell table
    if not all([x in cell_table.columns for x in BASE_COLS]):
        raise ValueError(
            "cell table must contain the following columns:{}".format(
                BASE_COLS))

    # Check markers/clusters
    if all([markers is None, clusters is None]):
        raise ValueError("markers and clusters cannot both be None")
    if markers is not None:
        if isinstance(markers, list) and len(markers) == 0:
            raise ValueError("list of marker names cannot be empty")
        if not isinstance(markers, list) or not all(
                [isinstance(x, str) for x in markers]):
            raise TypeError("markers must be a list of strings")
        if not all([x in cell_table.columns for x in markers]):
            raise ValueError("all markers must have a column in cell table")
    if clusters is not None:
        if isinstance(clusters, list) and len(clusters) == 0:
            raise ValueError("list of cluster ids cannot be empty")
        if not isinstance(clusters, list) or not all(
                [isinstance(x, int) for x in clusters]):
            raise TypeError("clusters must be a list of integers")


def check_featurize_cell_table_args(cell_table, feature_by, radius, cell_index):
    """Checks the input arguments of the featurize_cell_table() function.

    Args:
        cell_table (dict):
            A dictionary whose elements are the correctly formatted
            pd.DataFrames for each field of view.
        feature_by (str):
            One of "cluster", "marker", "avg_marker", or "count".
        radius (int):
            Pixel radius corresponding to cellular neighborhood size.
        cell_index (str):
            Name of the column in each field of view pd.Dataframe indicating
            reference cells.

    """
    # Check valid data types
    if not isinstance(cell_table, dict):
        raise TypeError("cell_table should be of type 'dict'")
    if not isinstance(cell_table[1], pd.DataFrame):
        raise TypeError("cell_table should contain formatted pd.DataFrames")
    if not isinstance(feature_by, str):
        raise TypeError("feature_by should be of type 'str'")
    if not isinstance(radius, int):
        raise TypeError("radius should be of type 'int'")
    if not isinstance(cell_index, str):
        raise TypeError("cell_index should be of type 'str'")

    # Check valid data values
    if feature_by not in ["cluster", "marker", "avg_marker", "count"]:
        raise ValueError(
            "feature_by must be one of 'cluster', 'marker', 'avg_marker', "
            "'count'")
    if radius < 25:
        raise ValueError("radius must not be less than 25")
    if cell_index not in cell_table[1].columns:
        raise ValueError("cell_index must be a valid column")


def check_create_difference_matrices_args(cell_table, features, training,
                                          inference):
    """Checks the input arguments of the create_difference_matrices() function.

    Args:
        cell_table (dict):
            A dictionary whose elements are the correctly formatted
            pd.DataFrames for each field of view.
        features (dict, pd.DataFrame):
            A featurized cell table and training split. Specifically, this is
            the output from featurize_cell_table().
        training (bool):
            If True, create the difference matrix for running training
            algorithm.
        inference (bool):
             If True, create the difference matrix for running inference
             algorithm.

    """

    if not isinstance(cell_table, dict):
        raise TypeError("cell_table must be of type 'dict'")
    if not isinstance(features, dict):
        raise TypeError("features must be of type 'dict'")
    if not isinstance(cell_table[1], pd.DataFrame):
        raise TypeError("cell_table should contain formatted pd.DataFrames")
    if not isinstance(features["featurized_fovs"], pd.DataFrame):
        raise TypeError("features should contain featurized pd.DataFrames")
    if not training and not inference:
        raise ValueError(
            "One or both of 'training' or 'inference' must be True")
    if training and features["train_features"] is None:
        raise ValueError("train_features cannot be 'None'")
