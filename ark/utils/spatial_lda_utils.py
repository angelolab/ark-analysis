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
        if len(markers) == 0:
            raise ValueError("list of marker names cannot be empty")
        verify_in_list(markers=markers, cell_table_columns=cell_table.columns.to_list())
    if clusters is not None:
        if len(clusters) == 0:
            raise ValueError("list of cluster ids cannot be empty")
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
