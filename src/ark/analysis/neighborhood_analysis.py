import os
import warnings

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import anndata
from tqdm.notebook import tqdm
from alpineer import misc_utils, io_utils

import ark.settings as settings
from ark.analysis import spatial_analysis_utils
from ark.utils.data_utils import load_anndatas


def create_neighborhood_matrix(anndata_dir, included_fovs=None, distlim=50,
                               self_neighbor=False, fov_col=settings.FOV_ID,
                               cell_label_col=settings.CELL_LABEL,
                               cell_type_col=settings.CELL_TYPE):
    """Calculates the number of neighbor phenotypes for each cell.

    Args:
        anndata_dir (str):
            path where the AnnData objects are stored.
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
        cell_type_col (str):
            column with the cell types.

    Returns:
        pandas.DataFrame:
            DataFrame containing phenotype counts per cell tupled with DataFrame containing
            phenotype frequencies of counts per phenotype/total phenotypes for each cell
    """

    # Set up input and parameters
    if included_fovs is None:
        included_fovs = io_utils.list_folders(anndata_dir, substrs=".zarr")
    else:
        included_fovs = [fov + '.zarr' for fov in included_fovs]

    # Check if included fovs found in fov_col
    misc_utils.verify_in_list(
        fov_names=included_fovs, unique_fovs=io_utils.list_folders(anndata_dir, substrs=".zarr"))

    # Load AnnData Collection and extract unique cell phenotypes
    fovs_ac = load_anndatas(anndata_dir=anndata_dir, join_obs="inner", join_obsm="inner")

    # Get the total number of phenotypes
    cluster_names = fovs_ac.obs[cell_type_col].unique()
    cluster_num = len(cluster_names)

    included_columns = [fov_col, cell_label_col, cell_type_col]

    # Initialize empty matrices for cell neighborhood data
    cell_neighbor_counts = pd.DataFrame(
        np.zeros((fovs_ac.obs.shape[0], cluster_num + len(included_columns)))
    )
    cell_neighbor_counts.index = fovs_ac.obs.index

    # Replace the first three columns of cell_neighbor_counts and rename cols
    cell_neighbor_counts[list(range(len(included_columns)))] = fovs_ac.obs[included_columns]
    cols = included_columns + list(cluster_names)
    cell_neighbor_counts.columns = cols

    cell_neighbor_freqs = cell_neighbor_counts.copy(deep=True)

    with tqdm(total=len(included_fovs), desc="Neighbors Matrix Generation", unit="FOVs") \
            as neighbor_mat_progress:
        for fov in included_fovs:
            neighbor_mat_progress.set_postfix(FOV=fov)

            # load in fov AnnData table
            fov_adata = anndata.read_zarr(
                os.path.join(anndata_dir, fov))
            fov_table = fov_adata.obs[included_columns]

            # Get the subset of phenotypes included in the current fov
            fov_cluster_names = fov_adata.obs[cell_type_col].unique()

            # Retrieve fov-specific distance matrix from AnnData table
            centroid_labels = list(fov_table.label)
            dist_matrix = fov_adata.obsp["distances"]
            dist_matrix = xr.DataArray(dist_matrix, coords=[centroid_labels, centroid_labels])

            # Get cell_neighbor_counts and cell_neighbor_freqs for fovs
            counts, freqs = spatial_analysis_utils.compute_neighbor_counts(
                fov_table, dist_matrix, distlim, self_neighbor,
                cell_label_col=cell_label_col, cluster_name_col=cell_type_col)

            # Add to neighbor counts+freqs for only matching phenos between fov and whole dataset
            cell_neighbor_counts.loc[fov_table.index, fov_cluster_names] = counts
            cell_neighbor_freqs.loc[fov_table.index, fov_cluster_names] = freqs

            # save neighbors matrix to fov AnnData table
            fov_adata.obsm[f"neighbors_counts_{cell_type_col}_{distlim}"] = \
                cell_neighbor_counts.loc[fov_table.index]
            fov_adata.obsm[f"neighbors_freqs_{cell_type_col}_{distlim}"] = \
                cell_neighbor_freqs.loc[fov_table.index]
            fov_adata.write_zarr(os.path.join(anndata_dir, fov), chunks=(1000, 1000))

            neighbor_mat_progress.update(1)

    # Remove cells that have no neighbors within the distlim
    total_cell_count = cell_neighbor_counts.shape[0]
    keep_cells = cell_neighbor_counts.drop(included_columns, axis=1).sum(axis=1) != 0
    cell_neighbor_counts = cell_neighbor_counts.loc[keep_cells].reset_index(drop=True)
    cell_neighbor_freqs = cell_neighbor_freqs.loc[keep_cells].reset_index(drop=True)

    # issue warning if more than 5% of cells are dropped
    if (cell_neighbor_counts.shape[0] / total_cell_count) < 0.95:
        warnings.warn(UserWarning("More than 5% of cells have no neighbor within the provided "
                                  "radius and have been omitted. We suggest increasing the "
                                  "distlim value to reduce the number of cells excluded from "
                                  "analysis."))

    return cell_neighbor_counts, cell_neighbor_freqs


def generate_cluster_matrix_results(all_data, neighbor_mat, cluster_num, seed=42,
                                    excluded_channels=None, included_fovs=None,
                                    cluster_label_col=settings.KMEANS_CLUSTER,
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
        seed (int):
            the random seed to set for k-means clustering
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
    neighbor_mat_data = neighbor_mat_data_all.drop([fov_col, label_col, cell_type_col], axis=1)

    # generate cluster labels
    cluster_labels = spatial_analysis_utils.generate_cluster_labels(
        neighbor_mat_data, cluster_num, seed=seed)

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
    channel_start = 0
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


def compute_cluster_metrics_inertia(neighbor_mat, min_k=2, max_k=10, seed=42,
                                    included_fovs=None, fov_col=settings.FOV_ID,
                                    label_col=settings.CELL_LABEL, cell_col=settings.CELL_TYPE):
    """Produce k-means clustering metrics to help identify optimal number of clusters using
       inertia

    Args:
        neighbor_mat (pandas.DataFrame):
            a neighborhood matrix, created from create_neighborhood_matrix
        min_k (int):
            the minimum k we want to generate cluster statistics for, must be at least 2
        max_k (int):
            the maximum k we want to generate cluster statistics for, must be at least 2
        seed (int):
            the random seed to set for k-means clustering
        included_fovs (list):
            fovs to include in analysis. If argument is none, default is all fovs used.
        fov_col (str):
            the name of the column in neighbor_mat indicating the fov
        label_col (str):
            the name of the column in neighbor_mat indicating the label
        cell_col (str):
            column with the cell phenotpype

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
    neighbor_mat_data = neighbor_mat_data.drop([fov_col, label_col, cell_col], axis=1)

    # generate the cluster score information
    neighbor_cluster_stats = spatial_analysis_utils.compute_kmeans_inertia(
        neighbor_mat_data=neighbor_mat_data, min_k=min_k, max_k=max_k, seed=seed)

    return neighbor_cluster_stats


def compute_cluster_metrics_silhouette(neighbor_mat, min_k=2, max_k=10, seed=42,
                                       included_fovs=None, fov_col=settings.FOV_ID,
                                       label_col=settings.CELL_LABEL, cell_col=settings.CELL_TYPE,
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
        seed (int):
            the random seed to set for k-means clustering
        included_fovs (list):
            fovs to include in analysis. If argument is none, default is all fovs used.
        fov_col (str):
            the name of the column in neighbor_mat indicating the fov
        label_col (str):
            the name of the column in neighbor_mat indicating the label
        cell_col (str):
            column with the cell phenotype
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
    neighbor_mat_data = neighbor_mat_data.drop([fov_col, label_col, cell_col], axis=1)

    # generate the cluster score information
    neighbor_cluster_stats = spatial_analysis_utils.compute_kmeans_silhouette(
        neighbor_mat_data=neighbor_mat_data, min_k=min_k, max_k=max_k,
        seed=seed, subsample=subsample
    )

    return neighbor_cluster_stats


def compute_cell_ratios(neighbors_mat, target_cells, reference_cells, fov_list, bin_number=10,
                        cell_col=settings.CELL_TYPE, fov_col=settings.FOV_ID,
                        label_col=settings.CELL_LABEL):
    """ Computes the target/reference and reference/target ratios for each FOV

    Args:
        neighbors_mat (pandas.DataFrame):
            a neighborhood matrix, created from create_neighborhood_matrix
        target_cells (list):
            invading cell phenotypes
        reference_cells (list):
            expected cell phenotypes
        fov_list (list):
            names of the fovs to compare
        bin_number (int):
            number of bins to use in histogram
        cell_col (str):
            column with the cell phenotype
        fov_col (str):
            column with the fovs
        label_col (str):
            column with the cell labels

    Returns:
        tuple(list, list):
            - the target/reference ratios of each FOV
            - the reference/target ratios of each FOV
    """

    targ_ref_ratio, ref_targ_ratio = np.empty(0), np.empty(0)
    for fov in fov_list:
        # subset neighbors mat by fov, drop fov name and labels
        neighbors_mat_fov = neighbors_mat[neighbors_mat[fov_col] == fov]
        misc_utils.verify_in_list(provided_column_names=[cell_col, fov_col, label_col],
                                  cell_neighbors_columns=neighbors_mat.columns)
        neighbors_mat_fov = neighbors_mat_fov.drop(columns=[fov_col, label_col])

        # get number of target and reference cells in sample
        target_total = neighbors_mat_fov[neighbors_mat_fov[cell_col].isin(target_cells)].shape[0]
        reference_total = neighbors_mat_fov[
            neighbors_mat_fov[cell_col].isin(reference_cells)].shape[0]

        if target_total == 0 or reference_total == 0:
            targ_ref_ratio = np.append(targ_ref_ratio, np.nan)
            ref_targ_ratio = np.append(ref_targ_ratio, np.nan)
        else:
            targ_ref_ratio = np.append(targ_ref_ratio, target_total / reference_total)
            ref_targ_ratio = np.append(ref_targ_ratio, reference_total / target_total)

    # remove nan values for plotting
    targ_ref_remove_nan = targ_ref_ratio[~np.isnan(targ_ref_ratio)]
    ref_targ_remove_nan = ref_targ_ratio[~np.isnan(ref_targ_ratio)]

    # filter values
    targ_ref_filter = targ_ref_remove_nan[targ_ref_remove_nan < 15]
    ref_targ_filter = ref_targ_remove_nan[ref_targ_remove_nan < 15]

    # create ratio plots
    sns.set(rc={'figure.figsize': (16, 4)})
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Population 1 / Population 2 Ratios")
    ax1.boxplot(targ_ref_filter, 0, 'c', vert=False)
    ax1.set(xlabel='Ratio')
    ax2.hist(targ_ref_filter, bins=bin_number)
    ax2.set(xlabel='Ratio', ylabel='Count')
    fig2, (ax3, ax4) = plt.subplots(1, 2)
    fig2.suptitle("Population 2 / Population 1 Ratios")
    ax3.boxplot(ref_targ_filter, 0, 'c', vert=False)
    ax3.set(xlabel='Ratio')
    ax4.hist(ref_targ_filter, bins=bin_number)
    ax4.set(xlabel='Ratio', ylabel='Count')

    ratio_data = pd.DataFrame(list(zip(fov_list, targ_ref_ratio)),
                              columns=['fov', 'cell_ratio'])
    return ratio_data


def compute_mixing_score(fov_neighbors_mat, target_cells, reference_cells, mixing_type,
                         ratio_threshold=5, cell_count_thresh=200,
                         cell_col=settings.CELL_TYPE, fov_col=settings.FOV_ID,
                         label_col=settings.CELL_LABEL):
    """ Compute and return the mixing score for the specified target/reference cell types

    Args:
        fov_neighbors_mat (pandas.DataFrame):
            a neighborhood matrix, created from create_neighborhood_matrix and subsetted for 1 fov
        target_cells (list):
            invading cell phenotypes
        reference_cells (list):
            expected cell phenotypes
        mixing_type (str):
            "homogeneous" or "percent", homogeneous is a symmetrical calculation
        ratio_threshold (int):
            maximum ratio of cell_types required to calculate a mixing score,
            under this labeled "cold"
        cell_count_thresh (int):
            minimum number of total cells from both populations to calculate a mixing score,
            under this labeled "cold"
        cell_col (str):
            column with the cell phenotype
        fov_col (str):
            column with the fovs
        label_col (str):
            column with the cell labels

    Returns:
        float:
            the mixing score for the FOV
    """

    # read in fov cell neighbors, drop fov, cell label, and cell type columns
    misc_utils.verify_in_list(provided_column_names=[cell_col, fov_col, label_col],
                              cell_neighbors_columns=fov_neighbors_mat.columns)
    fov_neighbors_mat = fov_neighbors_mat.drop(columns=[fov_col, label_col])

    # cell types validation
    overlap = [cell for cell in target_cells if cell in reference_cells]
    if overlap:
        raise ValueError(f"The following cell types were included in both the target and reference"
                         f" populations: {overlap}")
    all_cells = fov_neighbors_mat[cell_col].unique()

    # mixing_type validation
    if mixing_type not in ['percent', 'homogeneous']:
        raise ValueError(f'Please provide a valid mixing_type: "percent" or "homogeneous".')

    # get number of target and reference cells in sample
    target_total = fov_neighbors_mat[fov_neighbors_mat[cell_col].isin(target_cells)].shape[0]
    ref_total = fov_neighbors_mat[fov_neighbors_mat[cell_col].isin(reference_cells)].shape[0]

    # check cell count threshold
    if (target_total + ref_total) < cell_count_thresh:
        return np.nan, (target_total + ref_total)
    elif ref_total == 0 or target_total == 0:
        return np.nan, (target_total + ref_total)

    # check ratio threshold
    if ref_total/target_total > ratio_threshold or target_total/ref_total > ratio_threshold:
        return np.nan, (target_total + ref_total)

    # condense to total number of cell type interactions
    fov_neighbors_mat[cell_col] = fov_neighbors_mat[cell_col].replace(target_cells, 'target')
    fov_neighbors_mat[cell_col] = fov_neighbors_mat[cell_col].replace(reference_cells, 'reference')
    interactions_mat = fov_neighbors_mat.groupby(by=[cell_col]).sum(numeric_only=True)

    # combine cell interactions by target and reference populations
    interactions_mat['target'] = [0] * interactions_mat.shape[0]
    interactions_mat['reference'] = [0] * interactions_mat.shape[0]
    for target_cell in target_cells:
        if target_cell in all_cells:
            interactions_mat['target'] = interactions_mat['target'] + interactions_mat[target_cell]
    for reference_cell in reference_cells:
        if reference_cell in all_cells:
            interactions_mat['reference'] = interactions_mat['reference'] + \
                                            interactions_mat[reference_cell]

    # count interactions
    reference_target = interactions_mat.loc['target', 'reference']
    target_target = interactions_mat.loc['target', 'target']
    reference_reference = interactions_mat.loc['reference', 'reference']

    # mixing score calculation
    if mixing_type == 'percent':
        # percent mixing
        mixing_score = reference_target / (reference_target + target_target)
    elif mixing_type == 'homogeneous':
        # homogenous mixing
        mixing_score = reference_target / (target_target + reference_reference)

    return mixing_score, (target_total + ref_total)
