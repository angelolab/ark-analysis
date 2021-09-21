import numpy as np
from scipy.spatial.distance import pdist
from spatial_lda.visualization import plot_adjacency_graph

from ark.settings import BASE_COLS, CLUSTER_ID
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
    verify_in_list(required_columns=BASE_COLS, cell_table_columns=cell_table.columns.to_list())

    # Check markers/clusters
    if markers is None and clusters is None:
        raise ValueError("markers and clusters cannot both be None")
    if markers is not None:
        verify_in_list(markers=markers, cell_table_columns=cell_table.columns.to_list())
    if clusters is not None:
        cell_table_clusters = cell_table[CLUSTER_ID].unique().tolist()
        verify_in_list(clusters=clusters, cell_table_clusters=cell_table_clusters)


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

    verify_in_list(featurization=[featurization],
                   featurization_options=["cluster", "marker", "avg_marker", "count"])
    verify_in_list(cell_index=[cell_index], cell_table_columns=cell_table[1].columns.to_list())


def within_cluster_sums(data, labels):
    """Computes the pooled within-cluster sum of squares for the gap statistic .

        Args:
            data (pandas.DataFrame):
                A formatted and featurized cell table.
            labels (numpy.ndarray):
                A list of cluster labels corresponding to cluster assignments in data.

        Returns:
            float

            - The pooled within-cluster sum of squares for a given clustering iteration.
        """
    cluster_sums = []
    for x in np.unique(labels):
        d = data[labels == x]
        cluster_ss = pdist(d).sum() / (2 * d.shape[0])
        cluster_sums.append(cluster_ss)
    wk = np.sum(cluster_sums)
    return wk


def make_plot_fn(difference_matrices):
    """Helper function for making plots using the spatial-lda library.

            Args:
                difference_matrices (dict):
                    A dictionary of featurized difference matrices for each field of view.

            Returns:
                function

                - A function for plotting the adjacency network for each field of view..
            """

    def plot_fn(ax, sample_idx, features_df, fov_df):
        plot_adjacency_graph(ax, sample_idx, features_df, fov_df, difference_matrices)

    return plot_fn

