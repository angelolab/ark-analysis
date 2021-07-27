import os
import numpy as np
import pandas as pd

def check_format_cell_table_args(cell_table, markers, clusters, fovs):

    """
    Checks the input arguments of the format_cell_table() function.

    Args:
        cell_table (pd.DataFrame):
            A pandas DataFrame containing the columns of cell marker frequencies and/or cluster ids.
        markers (list, str):
            A list of strings corresponding to marker names.
        clusters (list, int):
            A list of integers corresponding to cluster ids.
        fovs (str, int):
            One of either "all_fovs" indicating all field of views are to be kept, or a list of integers corresponding
            to the index of each field of view which should be kept.

    Returns:
        None
    """

    base_cols = [
        "point",
        "label",
        "cell_size",
        "centroid-0",
        "centroid-1",
        "pixelfreq_hclust_cap",
        "name"
    ]

    # Check cell table
    if not isinstance(cell_table, pd.DataFrame):
        raise ValueError("cell_table must be a pd.DataFrame")

    if not all([True for x in base_cols if x in cell_table.columns]):
        raise ValueError("cell table must contain the following columns:{}".format(base_cols))

    # Check markers/clusters
    if all([markers is None, clusters is None]):
        raise ValueError("markers and clusters cannot both be None")
    if markers is not None:
        if isinstance(markers, list) and len(markers) == 0:
            raise ValueError("list of marker names cannot be empty")
        if not isinstance(markers, list) or not all([isinstance(x, str) for x in markers]):
            raise ValueError("clusters must be a list of integers")
        assert all([x in cell_table.columns for x in markers])
    if clusters is not None:
        if isinstance(clusters, list) and len(clusters) == 0:
            raise ValueError("list of cluster ids cannot be empty")
        if not isinstance(clusters, list) or not all([isinstance(x, int) for x in clusters]):
            raise ValueError("markers must be a list of strings")

    # Check fovs
    if fovs != "all_fovs":
        if not all([isinstance(x, int) for x in fovs]) or not isinstance(fovs, list) or len(fovs) == 0:
            raise ValueError("fovs must be 'all_fovs' or a list of integers")

    return None

