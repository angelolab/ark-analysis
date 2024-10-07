import os

import numpy as np
import pandas as pd
import scipy
import sklearn.metrics
import xarray as xr
from alpineer import io_utils, misc_utils
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
from tqdm.notebook import tqdm

import ark.settings as settings


def calc_dist_matrix(cell_table, save_path, fov_id=settings.FOV_ID, label_id=settings.CELL_LABEL,
                     centroid_ids=(settings.CENTROID_0, settings.CENTROID_1)):
    """Generate matrix of distances between center of pairs of cells.

    Saves each one individually to `save_path`.

    Args:
        cell_table (str):
            data frame with fov, label, and centroid information
        save_path (str):
            path to save the distance matrices
        fov_id (str):
            the column name containing the fov
        label_id (str):
            the column name containing the cell label
        centroid_ids (tuple):
            the columns identifying the centroids of each cell
    """

    # check that both label_dir and save_path exist
    io_utils.validate_paths([save_path])

    # load all the file names in label_dir
    fovs = cell_table[fov_id].unique()

    # iterate for each fov
    with tqdm(total=len(fovs), desc="Distance Matrix Generation", unit="FOVs") \
            as dist_mat_progress:
        for fov in fovs:
            dist_mat_progress.set_postfix(FOV=fov)

            fov_table = cell_table[cell_table[fov_id] == fov]

            # get centroid and label info
            centroids = [(row[centroid_ids[0]], row[centroid_ids[1]]) for indx, row in fov_table.iterrows()]
            centroid_labels = list(fov_table[label_id])

            # generate the distance matrix, then assign centroid_labels as coords
            dist_matrix = cdist(centroids, centroids).astype(np.float32)
            dist_mat_xarr = xr.DataArray(dist_matrix, coords=[centroid_labels, centroid_labels])

            # save the distance matrix to save_path
            dist_mat_xarr.to_netcdf(
                os.path.join(save_path, fov + '_dist_mat.xr'),
                format='NETCDF3_64BIT'
            )

            dist_mat_progress.update(1)


def append_distance_features_to_dataset(fov, dist_matrix, cell_table, distance_columns):
    """Appends selected distance features as 'cells' in distance matrix and cell table

    Args:
        fov (str):
            the name of the FOV
        dist_matrix (xarray.DataArray):
            a cells x cells matrix with the euclidian distance between centers of
            corresponding cells for the FOV
        cell_table (pd.DataFrame):
            Table of cell features. Must contain provided distance columns
        distance_columns (List[str]):
            List of column names which store feature distance.  These must exist in cell_table

    Returns:
        (pd.DataFrame, dict):
            Updated cell_table, and distance matricie indexed by fov name
    """

    misc_utils.verify_in_list(distance_columns=distance_columns, valid_columns=cell_table.columns)

    num_cell_types = max(list(cell_table[settings.CELL_TYPE].astype("category").cat.codes)) + 1
    dist_list = []

    fov_cells = cell_table.loc[cell_table[settings.FOV_ID] == fov]
    num_labels = max(fov_cells[settings.CELL_LABEL])
    for i, dist_col in enumerate(distance_columns):
        dist_list.append(pd.DataFrame([{
            settings.FOV_ID: fov,
            settings.CELL_LABEL: num_labels + i + 1,
            settings.CELL_TYPE: dist_col,
            settings.CELL_TYPE_NUM: num_cell_types + i + 1,
        }]))
        coords = (
            [max(dist_matrix.dim_0.values) + i + 1],
            fov_cells[settings.CELL_LABEL].values[:]
        )

        dist_matrix = xr.concat([dist_matrix, xr.DataArray(
            fov_cells[dist_col].values[np.newaxis, :], coords=coords
        )], dim='dim_0')

        dist_matrix = xr.concat([dist_matrix, xr.DataArray(
            fov_cells[dist_col].values[:, np.newaxis], coords=(coords[1], coords[0])
        )], dim='dim_1')

    distance_features = pd.concat(dist_list)
    cell_table = pd.concat([cell_table, distance_features])

    return cell_table, dist_matrix


def get_pos_cell_labels_channel(thresh, current_fov_channel_data, cell_labels, current_marker):
    """For channel enrichment, finds positive labels that match the current phenotype
    or identifies cells with positive expression values for the current marker
    (greater than the marker threshold).

    Args:
        thresh (int):
            current threshold for marker
        current_fov_channel_data (pandas.DataFrame):
            expression data for column markers for current patient
        cell_labels (pandas.DataFrame):
            the column of cell labels for current patient
        current_marker (str):
            the current marker that the positive labels are being found for

    Returns:
        list:
            List of all the positive labels"""

    # Subset only cells that are positive for the given marker
    marker1posinds = current_fov_channel_data[current_marker] > thresh
    # Get the cell labels of the positive cells
    mark1poslabels = cell_labels[marker1posinds]

    return mark1poslabels


def get_pos_cell_labels_cluster(pheno, current_fov_neighborhood_data,
                                cell_label_col, cell_type_col):
    """For cluster enrichment, finds positive labels that match the current phenotype
    or identifies cells with positive expression values for the current marker
    (greater than the marker threshold).

    Args:
        pheno (str):
            the current cell phenotype
        current_fov_neighborhood_data (pandas.DataFrame):
            data for the current patient
        cell_label_col (str):
            the name of the column indicating the cell label
        cell_type_col (str):
            the name of the column indicating the cell type

    Returns:
        list:
            List of all the positive labels"""

    # Subset only cells that are of the same phenotype
    pheno1posinds = current_fov_neighborhood_data[cell_type_col] == pheno
    # Get the cell labels of the cells of the phenotype
    mark1poslabels = current_fov_neighborhood_data.loc[:, cell_label_col][pheno1posinds]

    return mark1poslabels


def compute_close_cell_num(dist_mat, dist_lim, analysis_type,
                           current_fov_data=None, current_fov_channel_data=None,
                           cluster_ids=None, cell_types_analyze=None, thresh_vec=None,
                           cell_label_col=settings.CELL_LABEL,
                           cell_type_col=settings.CELL_TYPE_NUM):
    """Finds positive cell labels and creates matrix with counts for cells positive for
    corresponding markers. Computes close_num matrix for both Cell Label and Threshold spatial
    analyses.

    This function loops through all the included markers in the patient data and identifies cell
    labels positive for corresponding markers. It then subsets the distance matrix to only include
    these positive cells and records interactions based on whether cells are close to each other
    (within the dist_lim). It then stores the number of interactions in the index of close_num
    corresponding to both markers (for instance markers 1 and 2 would be in index [0, 1]).

    Args:
        dist_mat (numpy.ndarray):
            cells x cells matrix with the euclidian distance between centers of corresponding cells
        dist_lim (int):
            threshold for spatial enrichment distance proximity
        analysis_type (str):
            type of analysis, must be either cluster or channel
        current_fov_data (pandas.DataFrame):
            data for specific patient in expression matrix
        current_fov_channel_data (pandas.DataFrame):
            data of only column markers for Channel Analysis
        cluster_ids (numpy.ndarray):
            all the cell phenotypes in Cluster Analysis
        cell_types_analyze (list):
            a list of the cell types we wish to analyze, if None we set it equal to all cell types
        thresh_vec (numpy.ndarray):
            matrix of thresholds column for markers
        cell_label_col (str):
            the name of the column containing the cell labels
        cell_type_col (str):
            the name of the column containing the cell type numbers

    Returns:
        numpy.ndarray:
            2D array containing marker x marker matrix with counts for cells positive for
            corresponding markers, as well as a list of number of cell labels for marker 1
    """

    # assert our analysis type is valid
    good_analyses = ["cluster", "channel"]
    misc_utils.verify_in_list(analysis_type=analysis_type, good_analyses=good_analyses)

    # Initialize variables

    cell_labels = []

    # Subset data based on analysis type
    if analysis_type == "channel":
        # Subsetting the column with the cell labels
        cell_labels = current_fov_data[cell_label_col]

    # assign the dimension of close_num respective to type of analysis
    if analysis_type == "channel":
        num = len(thresh_vec)
    else:
        num = len(cluster_ids)

    # Create close_num, marker1_num, and marker2_num
    close_num = np.zeros((num, num), dtype=np.uint16)

    mark1_num = []
    mark1poslabels = []

    dist_mat_bin = xr.DataArray(
        ((dist_mat.values < dist_lim) & (dist_mat.values > 0)).astype(np.uint8),
        coords=dist_mat.coords
    )
    for j in range(num):
        if analysis_type == "cluster":
            mark1poslabels.append(
                get_pos_cell_labels_cluster(pheno=cluster_ids[j],
                                            current_fov_neighborhood_data=current_fov_data,
                                            cell_label_col=cell_label_col,
                                            cell_type_col=cell_type_col))
        else:
            mark1poslabels.append(
                get_pos_cell_labels_channel(thresh=thresh_vec[j],
                                            current_fov_channel_data=current_fov_channel_data,
                                            cell_labels=cell_labels,
                                            current_marker=current_fov_channel_data.columns[j]))
        mark1_num.append(len(mark1poslabels[j]))

    # iterating k from [j, end] cuts out 1/2 the steps (while symmetric)
    for j, m1n in enumerate(mark1_num):
        for k, m2n in enumerate(mark1_num[j:], j):
            dist_mat_bin_subset = dist_mat_bin.loc[
                mark1poslabels[j].values,
                mark1poslabels[k].values
            ].values
            count_close_num_hits = np.sum(dist_mat_bin_subset, dtype=np.uint16)

            close_num[j, k] = count_close_num_hits
            # symmetry :)
            close_num[k, j] = close_num[j, k]

    return close_num, mark1_num, mark1poslabels


def compute_neighbor_counts(current_fov_neighborhood_data, dist_matrix, distlim,
                            self_neighbor=False, cell_label_col=settings.CELL_LABEL,
                            cluster_name_col=settings.CELL_TYPE):
    """Calculates the number of neighbor phenotypes for each cell. The cell counts itself as a
    neighbor if self_neighbor=True.

    Args:
        current_fov_neighborhood_data (pandas.DataFrame):
            data for the current fov, including the cell labels, cell phenotypes, and cell
            phenotype
        dist_matrix (numpy.ndarray):
            cells x cells matrix with the euclidian distance between centers of corresponding cells
        distlim (int):
            threshold for distance proximity
        self_neighbor (bool):
            If true, cell counts itself as a neighbor in the analysis.
        cell_label_col (str):
            Column name with the cell labels
        cluster_name_col (str):
            Column name with the cell types
    Returns:
        tuple (pandas.DataFrame, pandas.DataFrame):
            - phenotype counts per cell
            - phenotype frequencies of counts per total for each cell
    """

    # subset our distance matrix based on the cell labels provided
    cell_labels = current_fov_neighborhood_data[cell_label_col].values
    cell_dist_mat = dist_matrix.loc[cell_labels, cell_labels].values

    # binarize distance matrix
    cell_dist_mat_bin = np.zeros(cell_dist_mat.shape)
    cell_dist_mat_bin[cell_dist_mat < distlim] = 1

    # remove cell as it's own neighbor
    if not self_neighbor:
        cell_dist_mat_bin[cell_dist_mat == 0] = 0

    # get num_neighbors for freqs
    num_neighbors = np.sum(cell_dist_mat_bin, axis=0)

    # create the 'phenotype has cell?' matrix, excluding non cell-label rows
    pheno_has_cell_pd = pd.get_dummies(current_fov_neighborhood_data.loc[:, cluster_name_col])
    pheno_names_from_tab = pheno_has_cell_pd.columns.values
    pheno_has_cell = pheno_has_cell_pd.to_numpy().T

    # dot binarized 'is neighbor?' matrix with pheno_has_cell to get counts
    counts = pheno_has_cell.dot(cell_dist_mat_bin).T
    counts_pd = pd.DataFrame(counts, columns=pheno_names_from_tab)
    counts_pd = counts_pd.set_index(current_fov_neighborhood_data.index.copy())

    # there may be errors if num_neighbors = 0, suppress these warnings
    np.seterr(invalid='ignore')

    # compute freqs with num_neighbors
    freqs = counts.T / num_neighbors
    freqs_pd = pd.DataFrame(freqs.T, columns=pheno_names_from_tab)
    freqs_pd = freqs_pd.set_index(current_fov_neighborhood_data.index.copy())
    # change na's to 0, will remove these cells before clustering
    freqs_pd = freqs_pd.fillna(0)

    return counts_pd, freqs_pd


def compute_kmeans_inertia(neighbor_mat_data, min_k=2, max_k=10, seed=42):
    """For a given neighborhood matrix, cluster and compute inertia using k-means clustering
       from the range of k=min_k to max_k

    Args:
        neighbor_mat_data (pandas.DataFrame):
            neighborhood matrix data with only the desired fovs
        min_k (int):
            the minimum k we want to generate cluster statistics for, must be at least 2
        max_k (int):
            the maximum k we want to generate cluster statistics for, must be at least 2
        seed (int):
            the random seed to set for k-means clustering

    Returns:
        xarray.DataArray:
            contains a single dimension, `cluster_num`, which indicates the inertia
            when `cluster_num` was set as k for k-means clustering
    """

    # create array we can store the results of each k for clustering
    coords = [np.arange(min_k, max_k + 1)]
    dims = ["cluster_num"]
    stats_raw_data = np.zeros(max_k - min_k + 1)
    cluster_stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    # iterate over each k value
    pb_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    for n in tqdm(range(min_k, max_k + 1), bar_format=pb_format):
        cluster_fit = KMeans(n_clusters=n, random_state=seed, n_init='auto').fit(neighbor_mat_data)
        cluster_stats.loc[n] = cluster_fit.inertia_

    return cluster_stats


def compute_kmeans_silhouette(neighbor_mat_data, min_k=2, max_k=10, seed=42, subsample=None):
    """For a given neighborhood matrix, cluster and compute Silhouette score using k-means
       from the range of k=min_k to max_k

    Args:
        neighbor_mat_data (pandas.DataFrame):
            neighborhood matrix data with only the desired fovs
        min_k (int):
            the minimum k we want to generate cluster statistics for, must be at least 2
        max_k (int):
            the maximum k we want to generate cluster statistics for, must be at least 2
        seed (int):
            the random seed to set for k-means clustering
        subsample (int):
            the number of cells that will be sampled from each neighborhood cluster for
            calculating Silhouette score
            If None, all cells will be used

    Returns:
        xarray.DataArray:
            contains a single dimension, `cluster_num`, which indicates the Silhouette score
            when `cluster_num` was set as k for k-means clustering
    """

    # create array we can store the results of each k for clustering
    coords = [np.arange(min_k, max_k + 1)]
    dims = ["cluster_num"]
    stats_raw_data = np.zeros(max_k - min_k + 1)
    cluster_stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    # iterate over each k value
    pb_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    for n in tqdm(range(min_k, max_k + 1), bar_format=pb_format):
        cluster_fit = KMeans(n_clusters=n, random_state=seed, n_init='auto').fit(neighbor_mat_data)
        cluster_labels = cluster_fit.labels_

        sub_dat = neighbor_mat_data.copy()
        sub_dat["cluster"] = cluster_labels

        if subsample is not None:
            # Subsample each cluster
            sub_dat = sub_dat.groupby("cluster").apply(
                lambda x: x.sample(
                    subsample, replace=len(x) < subsample, random_state=seed)
                ).reset_index(drop=True)

        cluster_score = sklearn.metrics.silhouette_score(sub_dat.drop("cluster", axis=1),
                                                         sub_dat["cluster"],
                                                         metric="euclidean")
        cluster_stats.loc[n] = cluster_score

    return cluster_stats


def generate_cluster_labels(neighbor_mat_data, cluster_num, seed=42):
    """Run k-means clustering with k=cluster_num

    Give the same data, given several runs the clusters will always be the same,
    but the labels assigned will likely be different

    Args:
        neighbor_mat_data (pandas.DataFrame):
            neighborhood matrix data with only the desired fovs
        cluster_num (int):
            the k we want to use when running k-means clustering
        seed (int):
            the random seed to set for k-means clustering

    Returns:
        numpy.ndarray:
            the neighborhood cluster labels assigned to each cell in neighbor_mat_data
    """

    cluster_fit = KMeans(n_clusters=cluster_num, random_state=seed, n_init=10).\
        fit(neighbor_mat_data)
    # Add 1 to avoid cluster number 0
    cluster_labels = cluster_fit.labels_ + 1

    return cluster_labels
