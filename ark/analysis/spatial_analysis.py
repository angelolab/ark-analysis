from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

import ark.settings as settings
from ark.utils import io_utils, load_utils, misc_utils, spatial_analysis_utils


def generate_channel_spatial_enrichment_stats(label_dir, marker_thresholds, all_data,
                                              suffix='_feature_0',
                                              xr_channel_name='segmentation_label', **kwargs):
    """Wrapper function for batching calls to `calculate_channel_spatial_enrichment` over fovs

    Args:
        label_dir (str | Pathlike):
            directory containing labeled tiffs
        marker_thresholds (numpy.ndarray):
            threshold values for positive marker expression
        all_data (pandas.DataFrame):
            data including fovs, cell labels, and cell expression matrix for all markers
        suffix (str):
            suffix for tiff file names
        xr_channel_name (str):
            channel name for label data array
        **kwargs (dict):
            args passed to `calculate_channel_spatial_enrichment`

    Returns:
        tuple (list, xarray.DataArray):

        - a list with each element consisting of a tuple of closenum and closenumrand for each
          fov included in the analysis
        - an xarray with dimensions (fovs, stats, num_channels, num_channels). The included
          stats variables for each fov are z, muhat, sigmahat, p, h, adj_p, and
          cluster_names
    """

    # parse files in label_dir
    all_label_names = io_utils.list_files(label_dir, substrs=['.tiff'])

    included_fovs = kwargs.get('included_fovs', None)
    if included_fovs:
        label_fovs = io_utils.extract_delimited_names(all_label_names, delimiter=suffix)
        all_label_names = \
            [all_label_names[i] for i, fov in enumerate(label_fovs) if fov in included_fovs]

    # create containers for batched return values
    values = []
    stats_datasets = []

    for label_name in tqdm(all_label_names, desc="Batch Completion", unit="batch"):
        label_maps = load_utils.load_imgs_from_dir(label_dir, files=[label_name],
                                                   xr_channel_names=[xr_channel_name],
                                                   trim_suffix=suffix)

        dist_mats = spatial_analysis_utils.calc_dist_matrix(label_maps)

        # filter 'included_fovs'
        if included_fovs:
            filtered_includes = set(dist_mats.keys()).intersection(included_fovs)
            kwargs['included_fovs'] = list(filtered_includes)

        batch_vals, batch_stats = \
            calculate_channel_spatial_enrichment(dist_mats, marker_thresholds, all_data, **kwargs)

        # append new values
        values = values + batch_vals

        # append new data array (easier than iteratively combining)
        stats_datasets.append(batch_stats)

    # combine list of data arrays into one
    stats = xr.concat(stats_datasets, dim="fovs")

    return values, stats


def calculate_channel_spatial_enrichment(dist_matrices_dict, marker_thresholds, all_data,
                                         excluded_channels=None, included_fovs=None,
                                         dist_lim=100, bootstrap_num=100,
                                         fov_col=settings.FOV_ID,
                                         cell_label_col=settings.CELL_LABEL, context_col=None):
    """Spatial enrichment analysis to find significant interactions between cells expressing
    different markers. Uses bootstrapping to permute cell labels randomly.

    Args:
        dist_matrices_dict (dict):
            contains a cells x cells matrix with the euclidian distance between centers of
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
        cell_label_col (str):
            cell label column name.
        context_col (str):
            column with context label.

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
        included_fovs = list(dist_matrices_dict.keys())
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

    # Only include the columns with the patient label, cell label, and cell phenotype
    if context_col is not None:
        context_names = all_data[context_col].unique()
        context_pairings = combinations_with_replacement(context_names, 2)

    for fov in included_fovs:
        # Subsetting expression matrix to only include patients with correct fov label
        current_fov_idx = all_data[fov_col] == fov
        current_fov_data = all_data[current_fov_idx]

        # Patients with correct label, and only columns of channel markers
        current_fov_channel_data = all_channel_data[current_fov_idx]

        # Retrieve fov-specific distance matrix from distance matrix dictionary
        dist_matrix = dist_matrices_dict[fov]

        # Get close_num and close_num_rand
        close_num, channel_nums, mark_pos_labels = spatial_analysis_utils.compute_close_cell_num(
            dist_mat=dist_matrix, dist_lim=dist_lim, analysis_type="channel",
            current_fov_data=current_fov_data, current_fov_channel_data=current_fov_channel_data,
            thresh_vec=thresh_vec, cell_label_col=cell_label_col)

        if context_col is not None:
            close_num_rand = np.zeros((*close_num.shape, bootstrap_num), dtype=np.uint16)

            context_nums_per_id = \
                current_fov_data.groupby(context_col)[cell_label_col].apply(list).to_dict()

            for name_i, name_j in context_pairings:
                # some FoVs may not have cells with a certain context, so they are skipped here
                try:
                    context_cell_labels = context_nums_per_id[name_i]
                    context_cell_labels.extend(context_nums_per_id[name_j])
                except KeyError:
                    continue
                context_cell_labels = np.unique(context_cell_labels)

                context_dist_mat = dist_matrix.loc[context_cell_labels, context_cell_labels]

                context_pos_labels = [
                    np.intersect1d(mark_pos_label, context_cell_labels)
                    for mark_pos_label in mark_pos_labels
                ]

                context_mark_nums = [len(cpl) for cpl in context_pos_labels]

                close_num_rand = close_num_rand + \
                    spatial_analysis_utils.compute_close_cell_num_random(
                        context_mark_nums, context_pos_labels, context_dist_mat, dist_lim,
                        bootstrap_num
                    )

        close_num_rand = spatial_analysis_utils.compute_close_cell_num_random(
            channel_nums, mark_pos_labels, dist_matrix, dist_lim, bootstrap_num)

        values.append((close_num, close_num_rand))

        # Get z, p, adj_p, muhat, sigmahat, and h
        stats_xr = spatial_analysis_utils.calculate_enrichment_stats(close_num, close_num_rand)
        stats.loc[fov, :, :] = stats_xr.values
    return values, stats


def generate_cluster_spatial_enrichment_stats(label_dir, all_data, suffix='_feature_0',
                                              xr_channel_name='segmentation_label', **kwargs):
    """ Wrapper function for batching calls to `calculate_cluster_spatial_enrichment` over fovs

    Args:
        label_dir (str | Pathlike):
            directory containing labeled tiffs
        all_data (pandas.DataFrame):
            data including fovs, cell labels, and cell expression matrix for all markers
        suffix (str):
            suffix for tiff file names
        xr_channel_name (str):
            channel name for label data array
        **kwargs (dict):
            args passed to `calculate_cluster_spatial_enrichment`

    Returns:
        tuple (list, xarray.DataArray):

        - a list with each element consisting of a tuple of closenum and closenumrand for each
          fov included in the analysis
        - an xarray with dimensions (fovs, stats, num_channels, num_channels). The included
          stats variables for each fov are z, muhat, sigmahat, p, h, adj_p, and
          cluster_names
    """

    # parse files in label_dir
    all_label_names = io_utils.list_files(label_dir, substrs=['.tiff'])

    included_fovs = kwargs.get('included_fovs', None)
    if included_fovs:
        label_fovs = io_utils.extract_delimited_names(all_label_names, delimiter=suffix)
        all_label_names = \
            [all_label_names[i] for i, fov in enumerate(label_fovs) if fov in included_fovs]

    # create containers for batched return values
    values = []
    stats_datasets = []

    for label_name in tqdm(all_label_names, desc="Batch Completion", unit="batch"):
        label_maps = load_utils.load_imgs_from_dir(label_dir, files=[label_name],
                                                   xr_channel_names=[xr_channel_name],
                                                   trim_suffix=suffix)

        dist_mats = spatial_analysis_utils.calc_dist_matrix(label_maps)

        # filter 'included_fovs'
        if included_fovs:
            filtered_includes = set(dist_mats.keys()).intersection(included_fovs)
            kwargs['included_fovs'] = list(filtered_includes)

        batch_vals, batch_stats = \
            calculate_cluster_spatial_enrichment(all_data, dist_mats, **kwargs)

        # append new values
        values = values + batch_vals

        # append new data array (easier than iteratively combining)
        stats_datasets.append(batch_stats)

    # combine list of data arrays into one
    stats = xr.concat(stats_datasets, dim="fovs")

    return values, stats


def calculate_cluster_spatial_enrichment(all_data, dist_matrices_dict, included_fovs=None,
                                         bootstrap_num=100, dist_lim=100, fov_col=settings.FOV_ID,
                                         cluster_name_col=settings.CELL_TYPE,
                                         cluster_id_col=settings.CELL_TYPE_NUM,
                                         cell_label_col=settings.CELL_LABEL, context_col=None,
                                         distance_cols=None):
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
            column with the cell phenotype number.
        cell_label_col (str):
            column with the cell labels.
        context_col (str):
            column with context labels. If None, no context is assumed.
        distance_cols (str):
            column names of feature distances to include in analysis.

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
        included_fovs = list(dist_matrices_dict.keys())
        num_fovs = len(included_fovs)
    else:
        num_fovs = len(included_fovs)

    values = []

    # check if included fovs found in fov_col
    misc_utils.verify_in_list(fov_names=included_fovs,
                              unique_fovs=all_data[fov_col].unique())

    all_data[cluster_id_col] = list(all_data[cluster_name_col].astype("category").cat.codes)
    if distance_cols:
        all_data, dist_matrices_dict = spatial_analysis_utils.append_distance_features_to_dataset(
            dist_matrices_dict, all_data, distance_cols
        )

    # Extract the names of the cell phenotypes
    cluster_names = all_data[cluster_name_col].drop_duplicates()
    # Extract the columns with the cell phenotype codes
    cluster_ids = all_data[cluster_id_col].drop_duplicates().values
    # Get the total number of phenotypes
    cluster_num = len(cluster_ids)

    # Only include the columns with the patient label, cell label, and cell phenotype
    all_pheno_data_cols = [fov_col, cell_label_col, cluster_id_col]
    all_pheno_data_cols += [] if context_col is None else [context_col]

    all_pheno_data = all_data[all_pheno_data_cols]

    if context_col is not None:
        context_names = all_data[context_col].unique()
        context_pairings = combinations_with_replacement(context_names, 2)

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
        close_num, pheno_nums, mark_pos_labels = spatial_analysis_utils.compute_close_cell_num(
            dist_mat=dist_mat, dist_lim=dist_lim, analysis_type="cluster",
            current_fov_data=current_fov_pheno_data, cluster_ids=cluster_ids,
            cell_label_col=cell_label_col)

        # subset distance matrix with context
        if context_col is not None:
            close_num_rand = np.zeros((*close_num.shape, bootstrap_num), dtype=np.uint16)

            context_nums_per_id = \
                current_fov_pheno_data.groupby(context_col)[cell_label_col].apply(list).to_dict()

            for name_i, name_j in context_pairings:
                # some FoVs may not have cells with a certain context, so they are skipped here
                try:
                    context_cell_labels = context_nums_per_id[name_i]
                    context_cell_labels.extend(context_nums_per_id[name_j])
                except KeyError:
                    continue

                context_cell_labels = np.unique(context_cell_labels)

                context_dist_mat = dist_mat.loc[context_cell_labels, context_cell_labels]

                context_pos_labels = [
                    np.intersect1d(mark_pos_label, context_cell_labels)
                    for mark_pos_label in mark_pos_labels
                ]

                context_pheno_nums = [len(cpl) for cpl in context_pos_labels]

                close_num_rand = close_num_rand + \
                    spatial_analysis_utils.compute_close_cell_num_random(
                        context_pheno_nums, context_pos_labels, context_dist_mat, dist_lim,
                        bootstrap_num
                    )

        else:
            close_num_rand = spatial_analysis_utils.compute_close_cell_num_random(
                pheno_nums, mark_pos_labels, dist_mat, dist_lim, bootstrap_num)

        values.append((close_num, close_num_rand))

        # Get z, p, adj_p, muhat, sigmahat, and h
        stats_xr = spatial_analysis_utils.calculate_enrichment_stats(close_num, close_num_rand)
        stats.loc[fov, :, :] = stats_xr.values

    return values, stats


def create_neighborhood_matrix(all_data, dist_matrices_dict, included_fovs=None, distlim=50,
                               self_neighbor=False, fov_col=settings.FOV_ID,
                               cell_label_col=settings.CELL_LABEL,
                               cluster_name_col=settings.CELL_TYPE):
    """Calculates the number of neighbor phenotypes for each cell.

    Args:
        all_data (pandas.DataFrame):
            data for all fovs. Includes the columns for fov, label, and cell phenotype.
        dist_matrices_dict (dict):
            contains a cells x cells centroid-distance matrix for every fov.  Keys are fov names
        included_fovs (list):
            fovs to include in analysis. If argument is none, default is all fovs used.
        distlim (int):
            cell proximity threshold. Default is 50.
        self_neighbor (bool):
            If true, cell counts itself as a neighbor in the analysis. Default is False.
        fov_col (str):
            column with the cell fovs.
        cell_label_col (str):
            column with the cell labels.
        cluster_name_col (str):
            column with the cell types.

    Returns:
        pandas.DataFrame:
            DataFrame containing phenotype counts per cell tupled with DataFrame containing
            phenotype frequencies of counts per phenotype/total phenotypes for each cell
    """

    # Set up input and parameters
    if included_fovs is None:
        included_fovs = all_data[fov_col].unique()

    # Check if included fovs found in fov_col
    misc_utils.verify_in_list(fov_names=included_fovs,
                              unique_fovs=all_data[fov_col].unique())

    # Subset just the fov, label, and cell phenotype columns
    all_neighborhood_data = all_data[[fov_col, cell_label_col, cluster_name_col]]
    # Extract the cell phenotypes
    cluster_names = all_neighborhood_data[cluster_name_col].drop_duplicates()
    # Get the total number of phenotypes
    cluster_num = len(cluster_names)

    # Initialize empty matrices for cell neighborhood data
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
            current_fov_neighborhood_data, dist_matrix, distlim, self_neighbor,
            cell_label_col=cell_label_col, cluster_name_col=cluster_name_col)

        # Add to neighbor counts + freqs for only the matching phenos between fov and whole dataset
        cell_neighbor_counts.loc[current_fov_neighborhood_data.index, fov_cluster_names] = counts
        cell_neighbor_freqs.loc[current_fov_neighborhood_data.index, fov_cluster_names] = freqs

    # Remove cells that have no neighbors within the distlim
    keep_cells = cell_neighbor_counts.drop([fov_col, cell_label_col], axis=1).sum(axis=1) != 0
    cell_neighbor_counts = cell_neighbor_counts.loc[keep_cells].reset_index(drop=True)
    cell_neighbor_freqs = cell_neighbor_freqs.loc[keep_cells].reset_index(drop=True)

    return cell_neighbor_counts, cell_neighbor_freqs


def generate_cluster_matrix_results(all_data, neighbor_mat, cluster_num, excluded_channels=None,
                                    included_fovs=None, cluster_label_col=settings.KMEANS_CLUSTER,
                                    fov_col=settings.FOV_ID, cell_type_col=settings.CELL_TYPE,
                                    label_col=settings.CELL_LABEL,
                                    pre_channel_col=settings.PRE_CHANNEL_COL,
                                    post_channel_col=settings.POST_CHANNEL_COL):
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
            fovs to include in analysis. If argument is None, default is all fovs used
        cluster_label_col (str):
            the name of the cluster label col we will create for neighborhood clusters
        fov_col (str):
            the name of the column in all_data and neighbor_mat indicating the fov
        cell_type_col (str):
            the name of the column in all_data indicating the cell type
        label_col (str):
            the name of the column in all_data indicating cell label
        pre_channel_col (str):
            the name of the column in all_data right before the first channel column
        post_channel_col (str):
            the name of the column in all_data right after the last channel column

    Returns:
        tuple (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame):

        - the expression matrix with the corresponding cluster labels attached,
          will only include fovs included in the analysis
        - an a x b count matrix (a = # of clusters, b = # of cell types) with
          cluster ids indexed row-wise and cell types indexed column-wise,
          indicates number of cell types that are within each cluster
        - an a x c mean matrix (a = # of clusters, c = # of markers) with
          cluster ids indexed row-wise and markers indexed column-wise,
          indicates the mean marker expression for each cluster id
    """

    # get fovs
    if included_fovs is None:
        included_fovs = neighbor_mat[fov_col].unique()

    # check if included fovs found in fov_col
    misc_utils.verify_in_list(fov_names=included_fovs,
                              unique_fovs=all_data[fov_col].unique())

    # check if all excluded column names found in all_data
    if excluded_channels is not None:
        misc_utils.verify_in_list(columns_to_exclude=excluded_channels,
                                  column_names=all_data.columns)

    # make sure number of clusters specified is valid
    if cluster_num < 2:
        raise ValueError("Invalid k provided for clustering")

    # subset neighbor mat
    neighbor_mat_data_all = neighbor_mat[neighbor_mat[fov_col].isin(included_fovs)]
    neighbor_mat_data = neighbor_mat_data_all.drop([fov_col, label_col], axis=1)

    # generate cluster labels
    cluster_labels = spatial_analysis_utils.generate_cluster_labels(
        neighbor_mat_data, cluster_num)

    # add labels to neighbor mat
    neighbor_mat_data_all[cluster_label_col] = cluster_labels

    # subset for data in cell table we want to keep
    all_data_clusters = all_data[all_data[fov_col].isin(included_fovs)]

    # combine with neighborhood data
    all_data_clusters = all_data_clusters.merge(
        neighbor_mat_data_all[[fov_col, label_col, cluster_label_col]], on=[fov_col, label_col])

    # create a count pivot table with cluster_label_col as row and cell_type_col as column
    group_by_cell_type = all_data_clusters.groupby(
        [cluster_label_col, cell_type_col]).size().reset_index(name="count")
    num_cell_type_per_cluster = group_by_cell_type.pivot(
        index=cluster_label_col, columns=cell_type_col, values="count").fillna(0).astype(int)

    # annotate index with "Cluster" to make visualization clear that these are cluster labels
    num_cell_type_per_cluster.index = ["Cluster" + str(c)
                                       for c in num_cell_type_per_cluster.index]

    # subsets the expression matrix to only have channel columns
    channel_start = np.where(all_data_clusters.columns == pre_channel_col)[0][0] + 1
    channel_end = np.where(all_data_clusters.columns == post_channel_col)[0][0]
    cluster_label_colnum = np.where(all_data_clusters.columns == cluster_label_col)[0][0]

    all_data_markers_clusters = \
        all_data_clusters.iloc[:, list(range(channel_start, channel_end)) + [cluster_label_colnum]]

    # drop excluded channels
    if excluded_channels is not None:
        all_data_markers_clusters = all_data_markers_clusters.drop(excluded_channels, axis=1)

    # create a mean pivot table with cluster_label_col as row and channels as column
    mean_marker_exp_per_cluster = all_data_markers_clusters.groupby([cluster_label_col]).mean()

    # annotate index with "Cluster" to make visualization clear that these are cluster labels
    mean_marker_exp_per_cluster.index = ["Cluster" + str(c)
                                         for c in mean_marker_exp_per_cluster.index]

    return all_data_clusters, num_cell_type_per_cluster, mean_marker_exp_per_cluster


def compute_cluster_metrics_inertia(neighbor_mat, min_k=2, max_k=10, included_fovs=None,
                                    fov_col=settings.FOV_ID, label_col=settings.CELL_LABEL):
    """Produce k-means clustering metrics to help identify optimal number of clusters using
       inertia

    Args:
        neighbor_mat (pandas.DataFrame):
            a neighborhood matrix, created from create_neighborhood_matrix
        min_k (int):
            the minimum k we want to generate cluster statistics for, must be at least 2
        max_k (int):
            the maximum k we want to generate cluster statistics for, must be at least 2
        included_fovs (list):
            fovs to include in analysis. If argument is none, default is all fovs used.
        fov_col (str):
            the name of the column in neighbor_mat indicating the fov
        label_col (str):
            the name of the column in neighbor_mat indicating the label

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
    if min_k < 2 or max_k < 2:
        raise ValueError("Invalid k provided for clustering")

    # check if included fovs found in fov_col
    misc_utils.verify_in_list(fov_names=included_fovs,
                              unique_fovs=neighbor_mat[fov_col].unique())

    # subset neighbor_mat accordingly, and drop the columns we don't need
    neighbor_mat_data = neighbor_mat[neighbor_mat[fov_col].isin(included_fovs)]
    neighbor_mat_data = neighbor_mat_data.drop([fov_col, label_col], axis=1)

    # generate the cluster score information
    neighbor_cluster_stats = spatial_analysis_utils.compute_kmeans_inertia(
        neighbor_mat_data=neighbor_mat_data, min_k=min_k, max_k=max_k)

    return neighbor_cluster_stats


def compute_cluster_metrics_silhouette(neighbor_mat, min_k=2, max_k=10, included_fovs=None,
                                       fov_col=settings.FOV_ID, label_col=settings.CELL_LABEL,
                                       subsample=None):
    """Produce k-means clustering metrics to help identify optimal number of clusters using
       Silhouette score

    Args:
        neighbor_mat (pandas.DataFrame):
            a neighborhood matrix, created from create_neighborhood_matrix
        min_k (int):
            the minimum k we want to generate cluster statistics for, must be at least 2
        max_k (int):
            the maximum k we want to generate cluster statistics for, must be at least 2
        included_fovs (list):
            fovs to include in analysis. If argument is none, default is all fovs used.
        fov_col (str):
            the name of the column in neighbor_mat indicating the fov
        label_col (str):
            the name of the column in neighbor_mat indicating the label
        subsample (int):
            the number of cells that will be sampled from each neighborhood cluster for
            calculating Silhouette score
            If None, all cells will be used

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
    if min_k < 2 or max_k < 2:
        raise ValueError("Invalid k provided for clustering")

    # check if included fovs found in fov_col
    misc_utils.verify_in_list(fov_names=included_fovs,
                              unique_fovs=neighbor_mat[fov_col].unique())

    # subset neighbor_mat accordingly, and drop the columns we don't need
    neighbor_mat_data = neighbor_mat[neighbor_mat[fov_col].isin(included_fovs)]
    neighbor_mat_data = neighbor_mat_data.drop([fov_col, label_col], axis=1)

    # generate the cluster score information
    neighbor_cluster_stats = spatial_analysis_utils.compute_kmeans_silhouette(
        neighbor_mat_data=neighbor_mat_data, min_k=min_k, max_k=max_k, subsample=subsample
    )

    return neighbor_cluster_stats
