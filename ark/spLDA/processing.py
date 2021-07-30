import pandas as pd
from sklearn.model_selection import train_test_split
from spatial_lda.featurization import *

from ark.settings import BASE_COLS
from ark.utils.spatial_lda_utils import check_format_cell_table_args, \
    check_featurize_cell_table_args

train_frac = 0.75


def format_cell_table(cell_table, markers=None, clusters=None):
    """
    Formats a cell table containing one for more fields of view to be
    compatible with the spatial_lda library.
    Args:
        cell_table (pd.DataFrame):
            A pandas DataFrame containing the columns of cell marker
            frequencies and/or cluster ids.
        markers (list, str):
            A list of strings corresponding to marker names.  Either markers
            or clusters must be provided.
        clusters (list, int):
            A list of integers corresponding to cluster ids.

    Returns:
        dict:

        - A dictionary of formatted cell tables for use in spatial-LDA
        analysis.  Each element in the dictionary is a
        pd.DataFrame corresponding to a single field of view.

    """

    # Check function arguments
    check_format_cell_table_args(cell_table=cell_table, markers=markers,
                                 clusters=clusters)

    # Only keep columns relevant for spatial-LDA
    if markers is not None:
        BASE_COLS.append(markers)
    drop_columns = [c for c in cell_table.columns if c not in BASE_COLS]
    cell_table = cell_table.drop(columns=drop_columns)

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
    fovs = np.unique(cell_table["point"])

    fov_dict = {}
    for i in fovs:
        df = cell_table[cell_table["point"] == i].drop(
            columns=["point", "label"])
        if clusters is not None:
            df = df[df["cluster_id"].isin(clusters)]
        df["is_index"] = True
        df["is_immune"] = True  # might remove this
        fov_dict[i] = df.reset_index(drop=True)

    # Save Arguments
    fov_dict["fovs"] = fovs
    fov_dict["markers"] = markers
    fov_dict["clusters"] = clusters

    return fov_dict


def featurize_cell_table(cell_table, feature_by="cluster", radius=100,
                         cell_index="is_index", n_processes=None,
                         train_split=True):
    """
    Args:
        cell_table (dict, pd.DataFrame):
            A formatted cell table for use in spatial-LDA analysis.
            Specifically, this is the output from
            format_cell_table().
        feature_by (str):
            One of four choices of featurization method:
                * marker: for each marker, count the total number of cells
                within a ``radius`` *r* from cell *i*
                having marker expression greater than 0.5.
                * avg_marker: for each marker, compute the average marker
                expression of all cells with a ``radius`` *r*
                from cell *i*.
                * cluster: for each cluster, count the total number of cells
                with a ``radius`` *r* from cell *i*
                belonging to that cell cluster.
                * count: counts the total number of cells within a ``radius``
                *r* from cell *i*.
        radius (int):
            Size of the radius, in pixels, used to featurize cellular
            neighborhoods.
        cell_index (str):
            Name of the column containing the reference cell indexes.  If not
            specified, all cells are used.
        n_processes (int):
            Number of parallel processes to use.
        train_split (bool):
            Create an index for training data in each field of view.

    Returns:
        dict:

        - A dictionary containing featurized cellular neighborhoods and
        training splits if applicable.
    """
    check_featurize_cell_table_args(cell_table=cell_table,
                                    feature_by=feature_by, radius=radius,
                                    cell_index=cell_index)
    # Define Featurization Function
    func_type = {
        "marker": neighborhood_to_marker,
        "cluster": neighborhood_to_cluster,
        "avg_marker": neighborhood_to_avg_marker,
        "count": neighborhood_to_count
    }
    if feature_by in ["marker", "avg_marker"]:
        neighborhood_feature_fn = functools.partial(func_type[feature_by],
                                                    markers=cell_table[
                                                        "markers"])
    else:
        neighborhood_feature_fn = functools.partial(func_type[feature_by])

    # Featurize FOVs
    feature_sample = {k: v for (k, v) in cell_table.items() if
                      k in cell_table["fovs"].tolist()}
    featurized_fovs = featurize_samples(feature_sample,
                                        neighborhood_feature_fn,
                                        radius=radius,
                                        is_anchor_col=cell_index,
                                        x_col='x',
                                        y_col='y',
                                        n_processes=n_processes,
                                        include_anchors=True)

    if train_split:
        all_sample_idxs = featurized_fovs.index.map(lambda x: x[0])
        _sets = train_test_split(featurized_fovs, test_size=1. - train_frac,
                                 stratify=all_sample_idxs)
        train_features_fraction, _ = _sets
    else:
        train_features_fraction = None

    return {"featurized_fovs": featurized_fovs,
            "train_features": train_features_fraction}


def create_difference_matrices(cell_table, features, training=True,
                               inference=True):
    """

    Args:
        cell_table (dict, pd.DataFrame):
            A formatted cell table for use in spatial-LDA analysis.
            Specifically, this is the output from
            format_cell_table().
        features (dict, pd.DataFrame):
            A featurized cell table and training split. Specifically, this is
            the output from featurize_cell_table().
        training (bool):
            If True, create the difference matrix for running training
            algorithm.  One or both of training and inference
            must be True.
        inference (bool):
             If True, create the difference matrix for running inference
             algorithm.

    Returns:
        dict:

        - A dictionary containing the training and inference difference
        matrices.

    """
    cell_table = {k: v for (k, v) in cell_table.items() if
                  k not in ["fovs", "markers", "clusters"]}
    # check args function here
    if training:
        train_diff_mat = make_merged_difference_matrices(
            sample_features=features["train_features"],
            sample_dfs=cell_table,
            x_col="x", y_col="y")
    else:
        train_diff_mat = None

    if inference:
        inference_diff_mat = make_merged_difference_matrices(
            sample_features=features["featurized_fovs"],
            sample_dfs=cell_table,
            x_col="x", y_col="y")
    else:
        inference_diff_mat = None

    return {"train_diff_mat": train_diff_mat,
            "inference_diff_mat": inference_diff_mat}
