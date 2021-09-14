import functools

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from spatial_lda import featurization as ft

import ark.utils.spatial_lda_utils as spu
from ark.settings import BASE_COLS


def format_cell_table(cell_table, markers=None, clusters=None):
    """Formats a cell table containing one for more fields of view to be
    compatible with the spatial_lda library.

    Args:
        cell_table (pandas.DataFrame):
            A DataFrame containing the columns of cell marker frequencies and/or cluster ids.
        markers (list):
            A list of strings corresponding to the markers in cell_table which will be used to
            train the spatial LDA model.  Either markers or clusters must be provided.
        clusters (list):
            A list of integers corresponding to cluster ids in cell_table which will be used to
            train the spatial LDA model.

    Returns:
        dict:

        - A dictionary of formatted cell tables for use in spatial-LDA analysis.  Each element
        in the dictionary is a Dataframe corresponding to a single field of view.
    """

    # Check function arguments
    spu.check_format_cell_table_args(cell_table=cell_table, markers=markers, clusters=clusters)

    # Only keep columns relevant for spatial-LDA
    if markers is not None:
        keep_cols = BASE_COLS + markers
    else:
        keep_cols = BASE_COLS
    drop_columns = [c for c in cell_table.columns if c not in keep_cols]
    cell_table_drop = cell_table.drop(columns=drop_columns)

    # Rename columns
    cell_table_drop = cell_table_drop.rename(
        columns={
            "centroid-0": "x",
            "centroid-1": "y",
            "FlowSOM_ID": "cluster_id",
            "cluster_labels": "cluster"
        })

    # Create dictionary of FOVs
    fovs = np.unique(cell_table_drop["SampleID"])

    fov_dict = {}
    for i in fovs:
        df = cell_table_drop[cell_table_drop["SampleID"] == i].drop(
            columns=["SampleID", "label"])
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


def featurize_cell_table(cell_table, featurization="cluster", radius=100, cell_index="is_index",
                         n_processes=None, train_frac=0.75):
    """Calculates statistics for local cellular neighborhoods based on the specified features
    and radius.

    Args:
        cell_table (dict):
            A formatted cell table for use in spatial-LDA analysis. Specifically, this is the
            output from :func:`~ark.spLDA.processing.format_cell_table`.
        featurization (str):
            One of four choices of featurization method, defaults to "cluster" if not provided:
                * marker: for each marker, count the total number of cells within a ``radius``
                *r* from cell *i* having marker expression greater than 0.5.
                * avg_marker: for each marker, compute the average marker expression of all
                cells within a ``radius`` *r* from cell *i*.
                * cluster: for each cluster, count the total number of cells within a ``radius``
                *r* from cell *i* belonging to that cell cluster.
                * count: counts the total number of cells within a ``radius`` *r* from cell *i*.
        radius (int):
            Size of the radius, in pixels, used to featurize cellular neighborhoods.
        cell_index (str):
            Name of the column containing the indexes of reference cells to be used in
            constructing local cellular neighborhoods.  If not specified, all cells are used.
        n_processes (int):
            Number of parallel processes to use.
        train_frac (float):
            The fraction of cells from each field of view to be extracted as training data.

    Returns:
        dict:

        - A dictionary containing a DataFrame of featurized cellular neighborhoods and a
        separate DataFrame for designated training data.
    """

    # Check arguments
    spu.check_featurize_cell_table_args(cell_table=cell_table, featurization=featurization,
                                        radius=radius, cell_index=cell_index)
    # Define Featurization Function
    func_type = {"marker": ft.neighborhood_to_marker, "cluster": ft.neighborhood_to_cluster,
                 "avg_marker": ft.neighborhood_to_avg_marker, "count": ft.neighborhood_to_count}

    if featurization in ["marker", "avg_marker"]:
        neighborhood_feature_fn = functools.partial(func_type[featurization],
                                                    markers=cell_table["markers"])
    else:
        neighborhood_feature_fn = functools.partial(func_type[featurization])

    # Featurize FOVs
    feature_sample = {k: v for (k, v) in cell_table.items() if k in cell_table["fovs"].tolist()}
    featurized_fovs = ft.featurize_samples(feature_sample,
                                           neighborhood_feature_fn,
                                           radius=radius,
                                           is_anchor_col=cell_index,
                                           x_col='x',
                                           y_col='y',
                                           n_processes=n_processes,
                                           include_anchors=True)
    # Extract training data sample
    all_sample_idxs = featurized_fovs.index.map(lambda x: x[0])
    train_features_fraction, _ = train_test_split(featurized_fovs, test_size=1. - train_frac,
                                                  stratify=all_sample_idxs)

    feature_dict = {"featurized_fovs": featurized_fovs, "train_features": train_features_fraction}
    return feature_dict


def create_difference_matrices(cell_table, features, training=True, inference=True):
    """Constructs the difference matrices used for training and inference for each field of view
    in the formatted cell table.

    Args:
        cell_table (dict):
            A formatted cell table for use in spatial-LDA analysis. Specifically, this is the
            output from :func:`~ark.spLDA.processing.format_cell_table`.
        features (dict):
            A dictionary containing the featurized cell table and the training data.
            Specifically, this is the output from
            :func:`~ark.spLDA.processing.featurize_cell_table`.
        training (bool):
            If True, create the difference matrix for running the training algorithm.  One or both
            of training and inference must be True.
        inference (bool):
             If True, create the difference matrix for running inference algorithm.

    Returns:
        dict:

        - A dictionary containing the difference matrices used for training and inference.
    """
    if not training and not inference:
        raise ValueError("One or both of 'training' or 'inference' must be True")

    cell_table = {
        k: v for (k, v) in cell_table.items() if k not in ["fovs", "markers", "clusters"]
    }
    # check args function here
    if training:
        train_diff_mat = ft.make_merged_difference_matrices(
            sample_features=features["train_features"], sample_dfs=cell_table,
            x_col="x", y_col="y")
    else:
        train_diff_mat = None

    if inference:
        inference_diff_mat = ft.make_merged_difference_matrices(
            sample_features=features["featurized_fovs"], sample_dfs=cell_table,
            x_col="x", y_col="y")
    else:
        inference_diff_mat = None

    matrix_dict = {"train_diff_mat": train_diff_mat, "inference_diff_mat": inference_diff_mat}
    return matrix_dict


def gap_stat(features, k, clust_inertia, num_boots=25):
    """Computes the Gap-statistic for a given k-means clustering model as introduced by
    Tibshirani, Walther and Hastie (2001).

    Args:
        features (pandas.DataFrame):
            A DataFrame of featurized cellular neighborhoods.  Specifically, this is one of the
            outputs of :func:`~ark.spLDA.processing.featurize_cell_table`.
        k (int):
            The number of clusters in the k-means model.
        clust_inertia (float):
            The calculated inertia from the k-means fit using the featurized data.
        num_boots (int):
            The number of bootstrap reference samples to generate.

    Returns:
        tuple (float, float)

        - Estimated difference between the the expected log within-cluster sum of squares and
        the observed log within-cluster sum of squares (a.k.a. the Gap-statistic).
        - A scaled estimate of the standard error of the expected log
        within-cluster sum of squares.

    """
    # Calculate the range of each feature column
    mins, maxs = features.apply(min, axis=0), features.apply(max, axis=0)
    n, p = features.shape
    w_kb = []
    # Create bootstrapped reference data
    boot_array = np.zeros((n, p))
    # Cluster each bootstrapped sample to get the inertia
    for b in range(num_boots):
        for i in range(p):
            boot_array[:, i] = np.random.uniform(low=mins[i], high=maxs[i], size=n)
        boot_clust = KMeans(n_clusters=k).fit(boot_array)
        w_kb.append(boot_clust.inertia_)
    # Gap statistic and standard error
    gap = np.log(w_kb).mean() - np.log(clust_inertia)
    s = np.log(w_kb).std() * np.sqrt(1 + 1 / num_boots)
    return gap, s


def compute_topic_eda(features, topics, num_boots=25):
    """Computes various metrics for k-means clustering models to help determine an
    appropriate number of topics for use in spatial-LDA analysis.

    Args:
        features (pandas.DataFrame):
            A DataFrame of featurized cellular neighborhoods.  Specifically, this is one of the
            outputs of :func:`~ark.spLDA.processing.featurize_cell_table`.
        topics (list):
            A list of integers corresponding to the different number of possible topics to
            investigate.
        num_boots (int):
            The number of bootstrap samples to use when calculating the Gap-statistic.

    Returns:
        dict:

        - A dictionary of dictionaries containing the corresponding metrics for each topic value
        provided.

    """
    # Check inputs
    if num_boots < 25:
        raise ValueError("Number of bootstrap samples must be at least 25")
    if min(topics) <= 2 or max(topics) >= features.shape[0] - 1:
        raise ValueError("Number of topics must be in [2, %d]" % (features.shape[0] - 1))

    stat_names = ['inertia', 'silhouette', 'gap_stat', 'gap_sds', 'percent_var_exp']
    stats = dict(zip(stat_names, [{} for name in stat_names]))

    # Compute the total sum of squared pairwise distances between all observations
    total_ss = np.sum(pdist(features) ** 2) / features.shape[0]
    for k in topics:
        cluster_fit = KMeans(n_clusters=k).fit(features)
        stats['inertia'][k] = cluster_fit.inertia_
        stats['silhouette'][k] = silhouette_score(features, cluster_fit.labels_, 'euclidean')
        stats['gap_stat'][k], stats['gap_sds'][k] = gap_stat(features, k, cluster_fit.inertia_,
                                                             num_boots)
        stats['percent_var_exp'][k] = (total_ss - cluster_fit.inertia_) / total_ss

    return stats


def fov_density(cell_table, total_pix=1024 ** 2):
    """Computes cellular density metrics for each field of view to determine an appropriate
    radius for the featurization step.

    Args:
        cell_table (dict):
            A formatted cell table for use in spatial-LDA analysis. Specifically, this is the
            output from :func:`~ark.spLDA.processing.format_cell_table`.
        total_pix (int):
            The total number of pixels in each field of view.

    Returns:
        dict:

        - A dictionary containing the average cell size and the cellular density for each field
        of view.  Cellular density is calculated by summing the total number of pixels occupied
        by cells divided by the total number of pixels in each field of view.

    """
    average_area = {}
    cellular_density = {}
    for i in range(1, 5):
        average_area[i] = cell_table[i].cell_size.mean()
        cellular_density[i] = np.sum(cell_table[i].cell_size) / total_pix

    density_stats = {"average_area": average_area, "cellular_density": cellular_density}

    return density_stats
