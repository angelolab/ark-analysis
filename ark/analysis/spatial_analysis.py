import pandas as pd
import xarray as xr
import numpy as np
from ark.utils import spatial_analysis_utils
from ark.utils import misc_utils

import ark.settings as settings


def calculate_channel_spatial_enrichment(dist_matrices_dict, marker_thresholds, all_data,
                                         excluded_channels=None, included_fovs=None,
                                         dist_lim=100, bootstrap_num=1000,
                                         fov_col=settings.FOV_ID):
    """Spatial enrichment analysis to find significant interactions between cells expressing
    different markers. Uses bootstrapping to permute cell labels randomly.

    Args:
        dist_matrices_dict (dict):
            Contains a cells x cells matrix with the euclidian distance between centers of
            corresponding cells for every fov
        marker_thresholds (numpy.ndarray):
            threshold values for positive marker expression
        all_data (pandas.DataFrame):
            data including fovs, cell labels, and cell expression matrix for all markers
        excluded_channels (list):
            channels to be excluded from the analysis.  Default is None.
        included_fovs (list):
            patient labels to include in analysis. If argument is none, default is all labels used.
        dist_lim (int):
            cell proximity threshold. Default is 100.
        bootstrap_num (int):
            number of permutations for bootstrap. Default is 1000.
        fov_col (str):
            column with the cell fovs.

    Returns:
        tuple (list, xarray.DataArray):

        - a list with each element consisting of a tuple of closenum and closenumrand for each
          fov included in the analysis
        - an xarray with dimensions (fovs, stats, num_channels, num_channels). The included
          stats variables for each fov are z, muhat, sigmahat, p, h, adj_p, and
          cluster_names
    """

    # Setup input and parameters
    if included_fovs is None:
        included_fovs = all_data[fov_col].unique()
        num_fovs = len(included_fovs)
    else:
        num_fovs = len(included_fovs)

    values = []

    # check if included fovs found in fov_col
    misc_utils.verify_in_list(fov_names=included_fovs,
                              unique_fovs=all_data[fov_col].unique())

    # check if all excluded column names found in all_data
    misc_utils.verify_in_list(columns_to_exclude=excluded_channels,
                              column_names=all_data.columns)

    # Subsets the expression matrix to only have channel columns
    channel_start = np.where(all_data.columns == settings.PRE_CHANNEL_COL)[0][0] + 1
    channel_end = np.where(all_data.columns == settings.POST_CHANNEL_COL)[0][0]

    all_channel_data = all_data.iloc[:, channel_start:channel_end]
    all_channel_data = all_channel_data.drop(excluded_channels, axis=1)

    # check that the markers are the same in marker_thresholdsa and all_channel_data
    misc_utils.verify_same_elements(markers_to_threshold=marker_thresholds.iloc[:, 0].values,
                                    all_markers=all_channel_data.columns.values)

    # reorder all_channel_data's marker columns the same as they appear in marker_thresholds
    all_channel_data = all_channel_data[marker_thresholds.iloc[:, 0].values]
    # List of all channels
    channel_titles = all_channel_data.columns
    # Length of channels list
    channel_num = len(channel_titles)

    # Create stats Xarray with the dimensions (fovs, stats variables, num_channels, num_channels)
    stats_raw_data = np.zeros((num_fovs, 7, channel_num, channel_num))
    coords = [included_fovs, ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"],
              channel_titles, channel_titles]
    dims = ["fovs", "stats", "marker1", "marker2"]
    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    # Subsetting threshold matrix to only include column with threshold values
    thresh_vec = marker_thresholds.iloc[:, 1].values

    for fov in included_fovs:
        # Subsetting expression matrix to only include patients with correct fov label
        current_fov_idx = all_data[fov_col] == fov
        current_fov_data = all_data[current_fov_idx]

        # Patients with correct label, and only columns of channel markers
        current_fov_channel_data = all_channel_data[current_fov_idx]

        # Retrieve fov-specific distance matrix from distance matrix dictionary
        dist_matrix = dist_matrices_dict[fov]

        # Get close_num and close_num_rand
        close_num, channel_nums, _ = spatial_analysis_utils.compute_close_cell_num(
            dist_mat=dist_matrix, dist_lim=100, analysis_type="channel",
            current_fov_data=current_fov_data, current_fov_channel_data=current_fov_channel_data,
            thresh_vec=thresh_vec)

        close_num_rand = spatial_analysis_utils.compute_close_cell_num_random(
            channel_nums, dist_matrix, dist_lim, bootstrap_num)

        values.append((close_num, close_num_rand))

        # Get z, p, adj_p, muhat, sigmahat, and h
        stats_xr = spatial_analysis_utils.calculate_enrichment_stats(close_num, close_num_rand)
        stats.loc[fov, :, :] = stats_xr.values
    return values, stats


def calculate_cluster_spatial_enrichment(all_data, dist_matrices_dict, included_fovs=None,
                                         bootstrap_num=1000, dist_lim=100, fov_col=settings.FOV_ID,
                                         cluster_name_col=settings.CELL_TYPE,
                                         cluster_id_col=settings.CLUSTER_ID,
                                         cell_label_col=settings.CELL_LABEL, context_labels=None):
    """Spatial enrichment analysis based on cell phenotypes to find significant interactions
    between different cell types, looking for both positive and negative enrichment. Uses
    bootstrapping to permute cell labels randomly.

    Args:
        all_data (pandas.DataFrame):
            data including fovs, cell labels, and cell expression matrix for all markers
        dist_matrices_dict (dict):
            A dictionary that contains a cells x cells matrix with the euclidian distance between
            centers of corresponding cells for every fov
        included_fovs (list):
            patient labels to include in analysis. If argument is none, default is all labels used
        bootstrap_num (int):
            number of permutations for bootstrap. Default is 1000
        dist_lim (int):
            cell proximity threshold. Default is 100
        fov_col (str):
            column with the cell fovs.
        cluster_name_col (str):
            column with the cell types.
        cluster_id_col (str):
            column with the cell phenotype IDs.
        cell_label_col (str):
            column with the cell labels.
        context_labels (dict):
            A dict that contains which specific types of cells we want to consider.
            If argument is None, we will not run context-dependent spatial analysis

    Returns:
        tuple (list, xarray.DataArray):

        - a list with each element consisting of a tuple of closenum and closenumrand for each
          fov included in the analysis
        - an xarray with dimensions (fovs, stats, number of channels, number of channels). The
          included stats variables for each fov are: z, muhat, sigmahat, p, h, adj_p, and
          cluster_names
    """

    # Setup input and parameters
    if included_fovs is None:
        included_fovs = all_data[fov_col].unique()
        num_fovs = len(included_fovs)
    else:
        num_fovs = len(included_fovs)

    values = []

    # check if included fovs found in fov_col
    misc_utils.verify_in_list(fov_names=included_fovs,
                              unique_fovs=all_data[fov_col].unique())

    # Extract the names of the cell phenotypes
    cluster_names = all_data[cluster_name_col].drop_duplicates()
    # Extract the columns with the cell phenotype codes
    cluster_ids = all_data[cluster_id_col].drop_duplicates().values
    # Get the total number of phenotypes
    cluster_num = len(cluster_ids)

    # Only include the columns with the patient label, cell label, and cell phenotype
    all_pheno_data = all_data[[fov_col, cell_label_col, cluster_id_col]]

    # Create stats Xarray with the dimensions (fovs, stats variables, num_markers, num_markers)
    stats_raw_data = np.zeros((num_fovs, 7, cluster_num, cluster_num))
    coords = [included_fovs, ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"],
              cluster_names, cluster_names]
    dims = ["fovs", "stats", "pheno1", "pheno2"]
    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    for fov in included_fovs:
        # Subsetting expression matrix to only include patients with correct fov label
        current_fov_idx = all_pheno_data.loc[:, fov_col] == fov
        current_fov_pheno_data = all_pheno_data[current_fov_idx]

        # Retrieve fov specific distance matrix from distance matrix dictionary
        dist_mat = dist_matrices_dict[fov]

        # Get close_num and close_num_rand
        close_num, pheno_nums, pheno_nums_per_id = spatial_analysis_utils.compute_close_cell_num(
            dist_mat=dist_mat, dist_lim=dist_lim, analysis_type="cluster",
            current_fov_data=current_fov_pheno_data, cluster_ids=cluster_ids)

        close_num_rand = spatial_analysis_utils.compute_close_cell_num_random(
            pheno_nums, dist_mat, dist_lim, bootstrap_num)

        # close_num_rand_context = spatial_analysis_utils.compute_close_cell_num_random(
        #     pheno_nums_per_id, dist_mat, dist_lim, bootstrap_num)

        values.append((close_num, close_num_rand))

        # Get z, p, adj_p, muhat, sigmahat, and h
        stats_xr = spatial_analysis_utils.calculate_enrichment_stats(close_num, close_num_rand)
        stats.loc[fov, :, :] = stats_xr.values

    return values, stats


def create_neighborhood_matrix(all_data, dist_matrices_dict, included_fovs=None, distlim=50,
                               fov_col=settings.FOV_ID, cluster_id_col=settings.CLUSTER_ID,
                               cell_label_col=settings.CELL_LABEL,
                               cluster_name_col=settings.CELL_TYPE):
    """Calculates the number of neighbor phenotypes for each cell.

    Args:
        all_data (pandas.DataFrame):
            data for all fovs. Includes the columns SampleID (fovs), cellLabelInImage (the cell
            label), FlowSOM_ID (the cell phenotype id)
        dist_matrices_dict (dict):
            Contains a cells x cells centroid-distance matrix for every fov.  Keys are fov names
        included_fovs (list):
            patient labels to include in analysis. If argument is none, default is all labels used.
        distlim (int):
            cell proximity threshold. Default is 50.
        fov_col (str):
            column with the cell fovs.
        cluster_id_col (str):
            column with the cell phenotype IDs.
        cell_label_col (str):
            column with the cell labels.
        cluster_name_col (str):
            column with the cell types.

    Returns:
        pandas.DataFrame:
            DataFrame containing phenotype counts per cell tupled with DataFrame containing
            phenotype frequencies of counts per phenotype/total phenotypes for each cell
    """

    # Setup input and parameters
    if included_fovs is None:
        included_fovs = all_data[fov_col].unique()

    # check if included fovs found in fov_col
    misc_utils.verify_in_list(fov_names=included_fovs,
                              unique_fovs=all_data[fov_col].unique())

    # Get the phenotypes
    cluster_names = all_data[cluster_name_col].drop_duplicates()

    # Subset just the sampleID, cellLabelInImage, and FlowSOMID, and cell phenotype
    all_neighborhood_data = all_data[[fov_col, cell_label_col, cluster_id_col, cluster_name_col]]
    # Extract the columns with the cell phenotype codes
    cluster_ids = all_neighborhood_data[cluster_id_col].drop_duplicates()
    # Get the total number of phenotypes
    cluster_num = len(cluster_ids)

    # initiate empty matrices for cell neighborhood data
    cell_neighbor_counts = pd.DataFrame(
        np.zeros((all_neighborhood_data.shape[0], cluster_num + 2))
    )

    # Replace the first, second columns of cell_neighbor_counts w/ fovs, cell-labels respectively
    cell_neighbor_counts[[0, 1]] = all_neighborhood_data[[fov_col, cell_label_col]]

    # Rename the columns to match cell phenotypes
    cols = [fov_col, cell_label_col] + list(cluster_names)
    cell_neighbor_counts.columns = cols

    cell_neighbor_freqs = cell_neighbor_counts.copy(deep=True)

    for fov in included_fovs:
        # Subsetting expression matrix to only include patients with correct fov label
        current_fov_idx = all_neighborhood_data.loc[:, fov_col] == fov
        current_fov_neighborhood_data = all_neighborhood_data[current_fov_idx]

        # Get the subset of phenotypes included in the current fov
        fov_cluster_names = current_fov_neighborhood_data[cluster_name_col].drop_duplicates()

        # Retrieve fov-specific distance matrix from distance matrix dictionary
        dist_matrix = dist_matrices_dict[fov]

        # Get cell_neighbor_counts and cell_neighbor_freqs for fovs
        counts, freqs = spatial_analysis_utils.compute_neighbor_counts(
            current_fov_neighborhood_data, dist_matrix, distlim)

        # add to neighbor counts + freqs for only the matching phenos between fov and whole dataset
        cell_neighbor_counts.loc[current_fov_neighborhood_data.index, fov_cluster_names] = counts
        cell_neighbor_freqs.loc[current_fov_neighborhood_data.index, fov_cluster_names] = freqs

    # drop label column, as this interferes with the neighborhood clustering step
    cell_neighbor_counts = cell_neighbor_counts.drop(columns=cell_label_col)
    cell_neighbor_freqs = cell_neighbor_freqs.drop(columns=cell_label_col)

    return cell_neighbor_counts, cell_neighbor_freqs


def generate_cluster_matrix_results(all_data, neighbor_mat, cluster_num, excluded_channels=None,
                                    included_fovs=None, cluster_label_col=settings.KMEANS_CLUSTER,
                                    fov_col=settings.FOV_ID, cell_type_col=settings.CELL_TYPE):
    """Generate the cluster info on all_data using k-means clustering on neighbor_mat.

    cluster_num has to be picked based on visualizations from compute_cluster_metrics.

    Args:
        all_data (pandas.DataFrame):
            data including fovs, cell labels, and cell expression matrix for all markers
        neighbor_mat (pandas.DataFrame):
            a neighborhood matrix, created from create_neighborhood_matrix
        cluster_num (int):
            the optimal k to pass into k-means clustering to generate the final clusters
            and corresponding results
        excluded_channels (list):
            all channel names to be excluded from analysis
        included_fovs (list):
            patient labels to include in analysis. If argument is None, default is all labels used
        cluster_label_col (str):
            the name of the cluster label col we will create
        fov_col (str):
            the name of the column in all_data and neighbor_mat determining the fov
        cell_type_col (str):
            the name of the column in all_data determining the cell type

    Returns:
        tuple (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame):

        - the expression matrix with the corresponding cluster labels attached
        - an a x b count matrix (a = # of clusters, b = # of cell types) with
          cluster ids indexed row-wise and cell types indexed column-wise,
          indicates number of cell types that are within each cluster
        - an a x c mean matrix (a = # of clusters, c = # of markers) with
          cluster ids indexed row-wise and markers indexed column-wise,
          indicates the mean marker expression for each cluster id
    """

    # error checking
    if included_fovs is None:
        included_fovs = neighbor_mat[fov_col].unique()

    # check if included fovs found in fov_col
    misc_utils.verify_in_list(fov_names=included_fovs,
                              unique_fovs=all_data[fov_col].unique())

    # check if all excluded column names found in all_data
    misc_utils.verify_in_list(columns_to_exclude=excluded_channels,
                              column_names=all_data.columns)

    # make sure number of clusters specified is valid
    if cluster_num < 2:
        raise ValueError("Invalid k provided for clustering")

    # subset neighbor mat
    neighbor_mat_data = neighbor_mat[neighbor_mat[fov_col].isin(included_fovs)]
    neighbor_mat_data = neighbor_mat_data.drop(fov_col, axis=1)

    # generate cluster labels
    cluster_labels = spatial_analysis_utils.generate_cluster_labels(
        neighbor_mat_data, cluster_num)

    all_data_clusters = all_data.copy()

    # add labels to all_data
    all_data_clusters[cluster_label_col] = cluster_labels

    # create a count pivot table with cluster_label_col as row and cell_type_col as column
    group_by_cell_type = all_data_clusters.groupby(
        [cluster_label_col, cell_type_col]).size().reset_index(name="count")
    num_cell_type_per_cluster = group_by_cell_type.pivot(
        index=cluster_label_col, columns=cell_type_col, values="count").fillna(0).astype(int)

    # annotate index with "Cluster" to make visualization clear that these are cluster labels
    num_cell_type_per_cluster.index = ["Cluster" + str(c)
                                       for c in num_cell_type_per_cluster.index]

    # Subsets the expression matrix to only have channel columns
    channel_start = np.where(all_data_clusters.columns == settings.PRE_CHANNEL_COL)[0][0] + 1
    channel_end = np.where(all_data_clusters.columns == settings.POST_CHANNEL_COL)[0][0]
    cluster_label_colnum = np.where(all_data_clusters.columns == cluster_label_col)[0][0]

    all_data_markers_clusters = \
        all_data_clusters.iloc[:, list(range(channel_start, channel_end)) + [cluster_label_colnum]]
    all_data_markers_clusters = all_data_markers_clusters.drop(excluded_channels, axis=1)

    # create a mean pivot table with cluster_label_col as row and channels as column
    mean_marker_exp_per_cluster = all_data_markers_clusters.groupby([cluster_label_col]).mean()

    # annotate index with "Cluster" to make visualization clear that these are cluster labels
    mean_marker_exp_per_cluster.index = ["Cluster" + str(c)
                                         for c in mean_marker_exp_per_cluster.index]

    return all_data_clusters, num_cell_type_per_cluster, mean_marker_exp_per_cluster


def compute_cluster_metrics(neighbor_mat, max_k=10, included_fovs=None,
                            fov_col='SampleID'):
    """Produce k-means clustering metrics to help identify optimal number of clusters

    Args:
        neighbor_mat (pandas.DataFrame):
            a neighborhood matrix, created from create_neighborhood_matrix
            the matrix should have the label col droppped
        max_k (int):
            the maximum k we want to generate cluster statistics for, must be at least 2
        included_fovs (list):
            patient labels to include in analysis. If argument is none, default is all labels used.
        fov_col (str):
            the name of the column in neighbor_mat determining the fov

    Returns:
        xarray.DataArray:
            an xarray with dimensions (num_k_values) where num_k_values is the range
            of integers from 2 to max_k included, contains the metric scores for each value
            in num_k_values
    """

    # set included_fovs to everything if not set
    if included_fovs is None:
        included_fovs = neighbor_mat[fov_col].unique()

    # make sure the user specifies a positive k
    if max_k < 2:
        raise ValueError("Invalid k provided for clustering")

    # check if included fovs found in fov_col
    misc_utils.verify_in_list(fov_names=included_fovs,
                              unique_fovs=neighbor_mat[fov_col].unique())

    # subset neighbor_mat accordingly, and drop the columns we don't need
    neighbor_mat_data = neighbor_mat[neighbor_mat[fov_col].isin(included_fovs)]
    neighbor_mat_data = neighbor_mat_data.drop(fov_col, axis=1)

    # generate the cluster score information
    neighbor_cluster_stats = spatial_analysis_utils.compute_kmeans_cluster_metric(
        neighbor_mat_data=neighbor_mat_data, max_k=max_k
    )

    return neighbor_cluster_stats
