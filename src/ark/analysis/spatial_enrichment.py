import os
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.notebook import tqdm
from alpineer import io_utils, load_utils, misc_utils

import ark.settings as settings
from ark.analysis import spatial_analysis_utils


def generate_channel_spatial_enrichment_stats(label_dir, dist_mat_dir, marker_thresholds, all_data,
                                              suffix='_whole_cell',
                                              xr_channel_name='segmentation_label', **kwargs):
    """Wrapper function for batching calls to `calculate_channel_spatial_enrichment` over fovs

    Args:
        label_dir (str | Pathlike):
            directory containing labeled tiffs
        dist_mat_dir (str | Pathlike):
            directory containing the distance matrices
        marker_thresholds (pd.DataFrame):
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

    # Validate paths
    io_utils.validate_paths([label_dir, dist_mat_dir])

    # parse files in label_dir
    all_label_names = io_utils.list_files(label_dir, substrs=[suffix + '.tiff'])

    included_fovs = kwargs.get('included_fovs', None)
    if included_fovs:
        label_fovs = io_utils.extract_delimited_names(all_label_names, delimiter=suffix)
        all_label_names = \
            [all_label_names[i] for i, fov in enumerate(label_fovs) if fov in included_fovs]

    # extra sanity check to ensure all_label_names fovs actually contained in all_dist_mat_fovs
    all_label_fovs = [os.path.splitext(f)[0].replace(suffix, '') for f in all_label_names]
    all_dist_mat_names = io_utils.list_files(dist_mat_dir, substrs=['.xr'])
    all_dist_mat_fovs = [f.replace('_dist_mat.xr', '') for f in all_dist_mat_names]

    misc_utils.verify_in_list(
        all_label_fovs=all_label_fovs,
        all_dist_mat_fovs=all_dist_mat_fovs
    )

    # pop the included_fovs key if specified, this won't be needed with a per-FOV iteration
    kwargs.pop('included_fovs', None)

    # create containers for batched return values
    values = []
    stats_datasets = []

    with tqdm(total=len(all_label_fovs), desc="Channel Spatial Enrichment") as chan_progress:
        for fov_name, label_file in zip(all_label_fovs, all_label_names):
            label_maps = load_utils.load_imgs_from_dir(label_dir, files=[label_file],
                                                       xr_channel_names=[xr_channel_name],
                                                       trim_suffix=suffix)
            dist_mat = xr.load_dataarray(os.path.join(dist_mat_dir, fov_name + '_dist_mat.xr'))

            batch_vals, batch_stats = \
                calculate_channel_spatial_enrichment(
                    fov_name, dist_mat, marker_thresholds, all_data, **kwargs
                )

            # append new values
            values = values + [batch_vals]

            # append new data array (easier than iteratively combining)
            stats_datasets.append(batch_stats)

            chan_progress.update(1)

    # combine list of data arrays into one
    stats = xr.concat(stats_datasets, dim="fovs")

    return values, stats


def calculate_channel_spatial_enrichment(fov, dist_matrix, marker_thresholds, all_data,
                                         excluded_channels=None,
                                         dist_lim=100, bootstrap_num=100,
                                         fov_col=settings.FOV_ID,
                                         cell_label_col=settings.CELL_LABEL, context_col=None):
    """Spatial enrichment analysis to find significant interactions between cells expressing
    different markers. Uses bootstrapping to permute cell labels randomly.

    Args:
        fov (str):
            the name of the FOV
        dist_matrix (xarray.DataArray):
            a cells x cells matrix with the euclidian distance between centers of
            corresponding cells for the FOV
        marker_thresholds (pd.DataFrame):
            threshold values for positive marker expression
        all_data (pandas.DataFrame):
            data including fovs, cell labels, and cell expression matrix for all markers
        excluded_channels (list):
            channels to be excluded from the analysis.  Default is None.
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
        tuple (tuple, xarray.DataArray):

        - a tuple of closenum and closenumrand for the fov computed in the analysis
        - an xarray with dimensions (fovs, stats, num_channels, num_channels). The included
          stats variables for each fov are z, muhat, sigmahat, p, h, adj_p, and
          cluster_names
    """

    # check if FOV found in fov_col
    misc_utils.verify_in_list(fov_name=[fov],
                              unique_fovs=all_data[fov_col].unique())

    # check if all excluded column names found in all_data
    if excluded_channels is not None:
        misc_utils.verify_in_list(columns_to_exclude=excluded_channels,
                                  column_names=all_data.columns)

    # Subsets the expression matrix to only have channel columns
    channel_start = np.where(all_data.columns == settings.PRE_CHANNEL_COL)[0][0] + 1
    channel_end = np.where(all_data.columns == settings.POST_CHANNEL_COL)[0][0]

    all_channel_data = all_data.iloc[:, channel_start:channel_end]
    if excluded_channels is not None:
        all_channel_data = all_channel_data.drop(excluded_channels, axis=1)
        marker_thresholds = marker_thresholds[~marker_thresholds["marker"].isin(excluded_channels)]

    # check that the markers are the same in marker_thresholds and all_channel_data
    misc_utils.verify_same_elements(markers_to_threshold=marker_thresholds.iloc[:, 0].values,
                                    all_markers=all_channel_data.columns.values)

    # reorder all_channel_data's marker columns the same as they appear in marker_thresholds
    all_channel_data = all_channel_data[marker_thresholds.iloc[:, 0].values]
    # List of all channels
    channel_titles = all_channel_data.columns
    # Length of channels list
    channel_num = len(channel_titles)

    # Create stats Xarray with the dimensions (fovs, stats variables, num_channels, num_channels)
    stats_raw_data = np.zeros((1, 7, channel_num, channel_num))
    coords = [[fov], ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"],
              channel_titles, channel_titles]
    dims = ["fovs", "stats", "marker1", "marker2"]
    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    # Subsetting threshold matrix to only include column with threshold values
    thresh_vec = marker_thresholds.iloc[:, 1].values

    # Only include the columns with the patient label, cell label, and cell phenotype
    if context_col is not None:
        context_names = all_data[context_col].unique()
        context_pairings = combinations_with_replacement(context_names, 2)

    # Subsetting expression matrix to only include patients with correct fov label
    current_fov_idx = all_data[fov_col] == fov
    current_fov_data = all_data[current_fov_idx]

    # Patients with correct label, and only columns of channel markers
    current_fov_channel_data = all_channel_data[current_fov_idx]

    # Get close_num and close_num_rand
    close_num, channel_nums, mark_pos_labels = spatial_analysis_utils.compute_close_cell_num(
        dist_mat=dist_matrix, dist_lim=dist_lim, analysis_type="channel",
        current_fov_data=current_fov_data, current_fov_channel_data=current_fov_channel_data,
        thresh_vec=thresh_vec, cell_label_col=cell_label_col
    )

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
    else:
        close_num_rand = spatial_analysis_utils.compute_close_cell_num_random(
            channel_nums, mark_pos_labels, dist_matrix, dist_lim, bootstrap_num
        )

    values = (close_num, close_num_rand)

    # Get z, p, adj_p, muhat, sigmahat, and h
    stats_xr = spatial_analysis_utils.calculate_enrichment_stats(close_num, close_num_rand)
    stats.loc[fov, :, :] = stats_xr.values

    return values, stats


def generate_cluster_spatial_enrichment_stats(label_dir, dist_mat_dir, all_data,
                                              suffix='_whole_cell',
                                              xr_channel_name='segmentation_label', **kwargs):
    """ Wrapper function for batching calls to `calculate_cluster_spatial_enrichment` over fovs

    Args:
        label_dir (str | Pathlike):
            directory containing labeled tiffs
        dist_mat_dir (str | Pathlike):
            directory containing the distance matrices
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
    all_label_names = io_utils.list_files(label_dir, substrs=[suffix + '.tiff'])

    included_fovs = kwargs.get('included_fovs', None)
    if included_fovs:
        label_fovs = io_utils.extract_delimited_names(all_label_names, delimiter=suffix)
        all_label_names = \
            [all_label_names[i] for i, fov in enumerate(label_fovs) if fov in included_fovs]

    # extra sanity check to ensure all_label_names fovs actually contained in all_dist_mat_fovs
    all_label_fovs = [os.path.splitext(f)[0].replace(suffix, '') for f in all_label_names]
    all_dist_mat_names = io_utils.list_files(dist_mat_dir, substrs=['.xr'])
    all_dist_mat_fovs = [f.replace('_dist_mat.xr', '') for f in all_dist_mat_names]

    misc_utils.verify_in_list(
        all_label_fovs=all_label_fovs,
        all_dist_mat_fovs=all_dist_mat_fovs
    )

    # pop the included_fovs key if specified, this won't be needed with a per-FOV iteration
    kwargs.pop('included_fovs', None)

    # create containers for batched return values
    values = []
    stats_datasets = []

    with tqdm(total=len(all_label_fovs), desc="Cluster Spatial Enrichment") as clust_progress:
        for fov_name, label_file in zip(all_label_fovs, all_label_names):
            label_maps = load_utils.load_imgs_from_dir(label_dir, files=[label_file],
                                                       xr_channel_names=[xr_channel_name],
                                                       trim_suffix=suffix)
            dist_mat = xr.load_dataarray(os.path.join(dist_mat_dir, fov_name + '_dist_mat.xr'))

            batch_vals, batch_stats = \
                calculate_cluster_spatial_enrichment(fov_name, all_data, dist_mat, **kwargs)

            # append new values
            values = values + [batch_vals]

            # append new data array (easier than iteratively combining)
            stats_datasets.append(batch_stats)

            clust_progress.update(1)

    # combine list of data arrays into one
    stats = xr.concat(stats_datasets, dim="fovs")

    return values, stats


def calculate_cluster_spatial_enrichment(fov, all_data, dist_matrix, included_fovs=None,
                                         bootstrap_num=100, dist_lim=100, fov_col=settings.FOV_ID,
                                         cluster_name_col=settings.CELL_TYPE,
                                         cluster_id_col=settings.CELL_TYPE_NUM,
                                         cell_label_col=settings.CELL_LABEL, context_col=None,
                                         distance_cols=None):
    """Spatial enrichment analysis based on cell phenotypes to find significant interactions
    between different cell types, looking for both positive and negative enrichment. Uses
    bootstrapping to permute cell labels randomly.

    Args:
        fov (str):
            the name of the FOV
        all_data (pandas.DataFrame):
            data including fovs, cell labels, and cell expression matrix for all markers
        dist_matrix (xarray.DataArray):
            a cells x cells matrix with the euclidian distance between centers of
            corresponding cells for the FOV
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
        tuple (tuple, xarray.DataArray):

        - a tuple of closenum and closenumrand for the fov computed in the analysis
        - an xarray with dimensions (fovs, stats, number of channels, number of channels). The
          included stats variables for each fov are: z, muhat, sigmahat, p, h, adj_p, and
          cluster_names
    """

    # check if FOV found in fov_col
    misc_utils.verify_in_list(fov_name=[fov],
                              unique_fovs=all_data[fov_col].unique())

    all_data[cluster_id_col] = list(all_data[cluster_name_col].astype("category").cat.codes)
    if distance_cols:
        all_data, dist_matrix = spatial_analysis_utils.append_distance_features_to_dataset(
            fov, dist_matrix, all_data, distance_cols
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
    stats_raw_data = np.zeros((1, 7, cluster_num, cluster_num))
    coords = [[fov], ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"],
              cluster_names, cluster_names]
    dims = ["fovs", "stats", "pheno1", "pheno2"]
    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    # Subsetting expression matrix to only include patients with correct fov label
    current_fov_idx = all_pheno_data.loc[:, fov_col] == fov
    current_fov_pheno_data = all_pheno_data[current_fov_idx]

    # Get close_num and close_num_rand
    close_num, pheno_nums, mark_pos_labels = spatial_analysis_utils.compute_close_cell_num(
        dist_mat=dist_matrix, dist_lim=dist_lim, analysis_type="cluster",
        current_fov_data=current_fov_pheno_data, cluster_ids=cluster_ids,
        cell_label_col=cell_label_col
    )

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

            context_dist_mat = dist_matrix.loc[context_cell_labels, context_cell_labels]

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
            pheno_nums, mark_pos_labels, dist_matrix, dist_lim, bootstrap_num
        )

    values = (close_num, close_num_rand)

    # Get z, p, adj_p, muhat, sigmahat, and h
    stats_xr = spatial_analysis_utils.calculate_enrichment_stats(close_num, close_num_rand)
    stats.loc[fov, :, :] = stats_xr.values

    return values, stats
