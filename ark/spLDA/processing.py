import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from ark.utils.spatial_lda_utils import check_format_cell_table_args
# from spatial_lda.featurization import *
# from spatial_lda.visualization import plot_adjacency_graph()


def format_cell_table(cell_table, markers=None, clusters=None, fovs="all_fovs"):
    """
    Formats a cell table containing one for more fields of view to be compatible with the spatial_lda library.
    Args:
        cell_table (pd.DataFrame):
            A pandas DataFrame containing the columns of cell marker frequencies and/or cluster ids.
        markers (list, str):
            A list of strings corresponding to marker names.
            Default: None
        clusters (list, int):
            A list of integers corresponding to cluster ids.
            Default: None
        fovs (str, int):
            One of either "all_fovs" indicating all field of views are to be kept, or a list of integers corresponding
            to the index of each field of view which should be kept.
            Default: "all_fovs"

    Returns:
        pd.DataFrame:

        - A formatted cell table for use in spatial-LDA analysis

    """

    # Check function arguments
    check_format_cell_table_args(cell_table=cell_table, markers=markers, clusters=clusters, fovs=fovs)

    # Only keep columns relevant for spatial-LDA
    all_columns = cell_table.columns
    keep_columns = [
        "point",
        "label",
        "cell_size",
        "centroid-0",
        "centroid-1",
        "pixelfreq_hclust_cap",
        "name"
    ]
    if markers is not None:
        keep_columns.append(markers)
    drop_columns = [c for c in all_columns if c not in keep_columns]
    cell_table = cell_table.drop(columns = drop_columns)

    # Rename columns
    cell_table = cell_table.rename(
        columns={
            "cell_size": "area",
            "centroid-0": "x",
            "centroid-1": "y",
            "pixelfreq_hclust_cap": "cluster_id",
            "name": "cluster"
        })

    # Create dictionary of FOVs
    if fovs == "all_fovs":
        fovs = [x + 1 for x in range(max(cell_table["point"]))]

    fov_dict = {}
    for i in fovs:
        df = cell_table[cell_table["point"] == i].drop(columns=["point", "label"])
        if clusters is not None:
            df = df[df["cluster_id"].isin(clusters)]
        df['is_index'] = True
        df['isimmune'] = True
        fov_dict[i] = df.reset_index(drop=True)

    return(fov_dict)








