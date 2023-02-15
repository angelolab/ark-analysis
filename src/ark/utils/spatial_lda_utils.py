import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import palettable.colorbrewer.qualitative as qual_palettes
import pandas as pd
import seaborn as sns
import spatial_lda.online_lda
from scipy.spatial.distance import pdist
from spatial_lda.visualization import _standardize_topics, plot_adjacency_graph
from alpineer import io_utils, misc_utils

from ark.settings import BASE_COLS, CELL_TYPE, LDA_PLOT_TYPES


def check_format_cell_table_args(cell_table, markers, clusters):
    """Checks the input arguments of the format_cell_table() function.

    Args:
        cell_table (pandas.DataFrame):
            A DataFrame containing the columns of cell marker frequencies and/or cluster ids.
        markers (list):
            A list of strings corresponding to marker names.
        clusters (list):
            A list of cell cluster names.
    """

    # Check cell table columns
    misc_utils.verify_in_list(required_columns=BASE_COLS,
                              cell_table_columns=cell_table.columns.to_list())

    # Check markers/clusters
    if markers is None and clusters is None:
        raise ValueError("markers and clusters cannot both be None")
    if markers is not None:
        misc_utils.verify_in_list(markers=markers, cell_table_columns=cell_table.columns.to_list())
    if clusters is not None:
        cell_table_clusters = cell_table[CELL_TYPE].unique().tolist()
        misc_utils.verify_in_list(clusters=clusters, cell_table_clusters=cell_table_clusters)


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
    # Check valid data types and values
    if not isinstance(radius, int):
        raise TypeError("radius should be of type 'int'")
    if radius < 25:
        raise ValueError("radius must not be less than 25")

    misc_utils.verify_in_list(featurization=[featurization],
                              featurization_options=["cluster", "marker", "avg_marker", "count"])

    if featurization in ["cluster"] and "clusters" not in cell_table:
        raise ValueError(
            "Cannot featurize clusters, because none were used for cell table formatting"
        )
    if featurization in ["marker", "avg_marker"] and "markers" not in cell_table:
        raise ValueError(
            "Cannot featurize markers, because none were used for cell table formatting"
        )

    key = list(cell_table.keys())[0]
    misc_utils.verify_in_list(cell_index=[cell_index],
                              cell_table_columns=cell_table[key].columns.to_list())


def within_cluster_sums(data, labels):
    """Computes the pooled within-cluster sum of squares for the gap statistic .

    Args:
        data (pandas.DataFrame):
            A formatted and featurized cell table.
        labels (numpy.ndarray):
            A list of cluster labels corresponding to cluster assignments in data.

    Returns:
        float:
            The pooled within-cluster sum of squares for a given clustering iteration.
    """
    cluster_sums = []
    for x in np.unique(labels):
        d = data[labels == x]
        cluster_ss = pdist(d).sum() / (2 * d.shape[0])
        cluster_sums.append(cluster_ss)
    wk = np.sum(cluster_sums)
    return wk


def plot_topics_heatmap(topics, features, normalizer=None, transpose=False, scale=0.4):
    """ Plots topic heatmap. Topics will be displayed on lower axis by default.

    Args:
        topics (pd.DataFrame | np.ndarray):
            topic assignments based off of trained featurization
        features (list | np.ndarray):
            feature names for display
        normalizer (Callable[(np.ndarray,), np.ndarray]):
            topic normalization for easier visualization. Default is standardization.
        transpose (bool):
            swap topic and features axes. helpful when the number of features is larger than the
            number of topics.
        scale (float):
            plot to text size scaling. for smaller text/larger label gaps, increase this value.
    """
    n_topics = topics.shape[0]
    if normalizer is not None:
        topics = normalizer(topics)
    else:
        topics = _standardize_topics(topics)

    topics = pd.DataFrame(topics, index=features,
                          columns=['Topic %d' % x for x in range(n_topics)])
    if transpose:
        topics = topics.T

    plt.subplots(figsize=(scale*topics.shape[1], scale*topics.shape[0]))
    sns.heatmap(topics, square=True, cmap='RdBu')


def plot_fovs_with_topics(ax, fov_idx, topic_weights, cell_table, uncolor_subset=None,
                          color_palette=qual_palettes.Set3_12.mpl_colors):
    """Helper function for plotting outputs from a fitted spatial-LDA model.

    Args:
        ax:
            Plot axis
        fov_idx (int):
            The index of the field of view to plot
        topic_weights (pandas.DataFrame):
            The data frame of cell topic weights from a fitted spatial-LDA model.
        cell_table (dict):
            A formatted cell table
        uncolor_subset (str | None):
            Name of cell type to leave uncolored
        color_palette (List[Tuple[float, float, float]]):
            Color palette in mpl format
    """
    colors = np.array(color_palette[:topic_weights.shape[1]])
    cell_coords = cell_table[fov_idx]
    cell_indices = topic_weights.index.map(lambda x: x[1])
    coords = cell_table[fov_idx].loc[cell_indices]
    if uncolor_subset is not None:
        immune_coords = cell_coords[cell_coords[uncolor_subset]]
        ax.scatter(immune_coords['y'], -immune_coords['x'],
                   s=5, c='k', label=uncolor_subset, alpha=0.1)

    ax.scatter(coords['y'], -coords['x'], s=2,
               c=colors[np.argmax(np.array(topic_weights), axis=1), :])
    ax.set_title(f"FOV {fov_idx}")
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)


def make_plot_fn(plot="adjacency", difference_matrices=None, topic_weights=None, cell_table=None,
                 color_palette=qual_palettes.Set3_12.mpl_colors):
    """Helper function for making plots using the spatial-lda library.

    Args:
        plot (str):
            Which plot function to return.  One of "adjacency" or "topic_assignment"
        difference_matrices (dict):
            A dictionary of featurized difference matrices for each field of view.
        topic_weights (pandas.DataFrame):
            The data frame of cell topic weights from a fitted spatial-LDA model.
        cell_table (dict):
            A formatted cell table
        color_palette (List[Tuple[float, float, float]]):
            Color palette in mpl format (list of rgb tuples)

    Returns:
        Callable:
            A function for plotting spatial-LDA data.
    """
    # check args
    misc_utils.verify_in_list(plot=[plot], plot_options=LDA_PLOT_TYPES)

    if plot == "adjacency":
        if difference_matrices is None:
            raise ValueError("Must provide difference_matrices")

        def plot_fn(ax, sample_idx, features_df, fov_df):
            plot_adjacency_graph(ax, sample_idx, features_df, fov_df, difference_matrices)
    else:
        if topic_weights is None or cell_table is None:
            raise ValueError("Must provide cell_table and topic_weights")

        def plot_fn(ax, sample_idx, features_df=topic_weights, fov_df=cell_table):
            plot_fovs_with_topics(ax, sample_idx, features_df, fov_df, color_palette=color_palette)

    return plot_fn


def save_spatial_lda_file(data, dir, file_name, format="pkl"):
    """Helper function saving spatial-LDA objects.

    Args:
        data (dict, pandas.DataFrame):
            A dictionary or data frame.
        dir (str):
            The directory where the data will be saved.
        file_name (str):
            Name of the data file.
        format (str):
            The designated file extension.  Must be one of either 'pkl' or 'csv'.
    """
    if not os.path.exists(dir):
        raise ValueError("'dir' must be a valid directory.")
    file_name += "." + format
    file_path = os.path.join(dir, file_name)

    if format == "pkl":
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    elif format == "csv":
        if type(data) == dict:
            raise ValueError("'data' is of type dict.  Use format='pkl' instead.")
        elif type(data) == spatial_lda.online_lda.LatentDirichletAllocation:
            raise ValueError("'data' is a spatial_lda model.  Use format='pkl' instead.")
        else:
            data.to_csv(file_path)
    else:
        raise ValueError("format must be either 'csv' or 'pkl'.")


def read_spatial_lda_file(dir, file_name, format="pkl"):
    """Helper function reading spatial-LDA objects.

    Args:
        dir (str):
            The directory where the data is located.
        file_name (str):
            Name of the data file.
        format (str):
            The designated file extension.  Must be one of either 'pkl' or 'csv'.

    Returns:
        pd.DataFrame | dict | spatial_lda.online_lda.LatentDirchletAllocation:
            Either an individual data frame, a dictionary, or a spatial_lda model.
    """
    file_name += "." + format
    file_path = os.path.join(dir, file_name)
    io_utils.validate_paths(file_path)

    if format == "pkl":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    elif format == "csv":
        data = pd.read_csv(file_path)
    else:
        raise ValueError("format must be either 'csv' or 'pkl'.")

    return data
