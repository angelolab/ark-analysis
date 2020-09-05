import pandas as pd
import xarray as xr
import numpy as np
from ark.utils import spatial_analysis_utils


def calculate_channel_spatial_enrichment(dist_matrices_dict, marker_thresholds, all_data,
                                         excluded_colnames=None, included_fovs=None,
                                         dist_lim=100, bootstrap_num=1000, fov_col="SampleID"):
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
        excluded_colnames (list):
            all column names that are not markers. If argument is none, default is
            ["cell_size", "Background", "HH3",
            "summed_channel", "label", "area",
            "eccentricity", "major_axis_length",
            "minor_axis_length", "perimeter", "fov"]
        included_fovs (list):
            patient labels to include in analysis. If argument is none, default is all labels used.
        dist_lim (int):
            cell proximity threshold. Default is 100.
        bootstrap_num (int):
            number of permutations for bootstrap. Default is 1000.
        fov_col (str):
            column with the cell fovs. Default is 'SampleID'

    Returns:
        tuple (list, xarray.DataArray):
            - a list with each element consisting of a tuple of closenum and closenumrand for each
              point included in the analysis
            - an xarray with dimensions (fovs, stats, num_channels, num_channels). The included
              stats variables for each point are z, muhat, sigmahat, p, h, adj_p, and
              cluster_names
    """

    # Setup input and parameters
    if included_fovs is None:
        included_fovs = list(set(all_data[fov_col]))
        num_fovs = len(included_fovs)
    else:
        num_fovs = len(included_fovs)

    values = []

    if excluded_colnames is None:
        excluded_colnames = ["cell_size", "Background", "HH3",
                             "summed_channel", "label", "area",
                             "eccentricity", "major_axis_length", "minor_axis_length",
                             "perimeter", "fov"]

    # Error Checking
    if not np.isin(excluded_colnames, all_data.columns).all():
        raise ValueError("Column names were not found in Expression Matrix")

    if not np.isin(included_fovs, all_data[fov_col]).all():
        raise ValueError("Fovs were not found in Expression Matrix")

    # Subsets the expression matrix to only have channel columns
    all_channel_data = all_data.drop(excluded_colnames, axis=1)
    # List of all channels
    channel_titles = all_channel_data.columns
    # Length of channels list
    channel_num = len(channel_titles)

    # Check to see if order of channel thresholds is same as in expression matrix
    if not (list(marker_thresholds.iloc[:, 0]) == channel_titles).any():
        raise ValueError("Threshold Markers do not match markers in Expression Matrix")

    # Create stats Xarray with the dimensions (fovs, stats variables, num_channels, num_channels)
    stats_raw_data = np.zeros((num_fovs, 7, channel_num, channel_num))
    coords = [included_fovs, ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"],
              channel_titles, channel_titles]
    dims = ["fovs", "stats", "marker1", "marker2"]
    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    # Subsetting threshold matrix to only include column with threshold values
    thresh_vec = marker_thresholds.iloc[:, 1]

    for i in range(0, len(included_fovs)):
        # Subsetting expression matrix to only include patients with correct fov label
        current_fov_idx = all_data[fov_col] == included_fovs[i]
        current_fov_data = all_data[current_fov_idx]
        # Patients with correct label, and only columns of channel markers
        current_fov_channel_data = all_channel_data[current_fov_idx]

        # Retrieve point specific distance matrix from distance matrix dictionary
        dist_matrix = dist_matrices_dict[included_fovs[i]]

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
        stats.loc[included_fovs[i], :, :] = stats_xr.values
    return values, stats


def calculate_cluster_spatial_enrichment(all_data, dist_matrices_dict, included_fovs=None,
                                         bootstrap_num=1000, dist_lim=100, fov_col="SampleID",
                                         cluster_name_col="cell_type", cluster_id_col="FlowSOM_ID",
                                         cell_label_col="cellLabelInImage", context_labels=None):
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
            column with the cell fovs. Default is 'SampleID'
        cluster_name_col (str):
            column with the cell types. Default is 'cell_type'
        cluster_id_col (str):
            column with the cell phenotype IDs. Default is 'FlowSOM_ID'
        cell_label_col (str):
            column with the cell labels. Default is 'cellLabelInImage'
        context_labels (dict):
            A dict that contains which specific types of cells we want to consider.
            If argument is None, we will not run context-dependent spatial analysis

    Returns:
        tuple (list, xarray.DataArray):
            - a list with each element consisting of a tuple of closenum and closenumrand for each
              point included in the analysis
            - an xarray with dimensions (fovs, stats, number of channels, number of channels). The
              included stats variables for each point are: z, muhat, sigmahat, p, h, adj_p, and
              cluster_names
    """

    # Setup input and parameters
    if included_fovs is None:
        included_fovs = list(set(all_data[fov_col]))
        num_fovs = len(included_fovs)
    else:
        num_fovs = len(included_fovs)

    values = []

    # Error Checking
    if not np.isin(included_fovs, all_data[fov_col]).all():
        raise ValueError("Fovs were not found in Expression Matrix")

    # Extract the names of the cell phenotypes
    cluster_names = all_data[cluster_name_col].drop_duplicates()
    # Extract the columns with the cell phenotype codes
    cluster_ids = all_data[cluster_id_col].drop_duplicates()
    # Get the total number of phenotypes
    cluster_num = len(cluster_ids)

    # Only include the columns with the patient label, cell label, and cell phenotype
    all_pheno_data = all_data[[fov_col, cell_label_col, cluster_id_col]]

    # Create stats Xarray with the dimensions (points, stats variables, num_markers, num_markers)
    stats_raw_data = np.zeros((num_fovs, 7, cluster_num, cluster_num))
    coords = [included_fovs, ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"],
              cluster_names, cluster_names]
    dims = ["fovs", "stats", "pheno1", "pheno2"]
    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    for i in range(0, len(included_fovs)):
        # Subsetting expression matrix to only include patients with correct fov label
        current_fov_idx = all_pheno_data.iloc[:, 0] == included_fovs[i]
        current_fov_pheno_data = all_pheno_data[current_fov_idx]

        # Retrieve point specific distance matrix from distance matrix dictionary
        dist_mat = dist_matrices_dict[included_fovs[i]]

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
        stats.loc[included_fovs[i], :, :] = stats_xr.values

    return values, stats


def create_neighborhood_matrix(all_data, dist_matrices_dict, included_fovs=None, distlim=50,
                               fov_col="SampleID", cluster_id_col="FlowSOM_ID",
                               cell_label_col="cellLabelInImage", cluster_name_col="cell_type"):
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
            column with the cell fovs. Default is 'SampleID'
        cluster_id_col (str):
            column with the cell phenotype IDs. Default is 'FlowSOM_ID'
        cell_label_col (str):
            column with the cell labels. Default is 'cellLabelInImage'
        cluster_name_col (str):
            column with the cell types. Default is 'cell_type'

    Returns:
        pandas.DataFrame:
            DataFrame containing phenotype counts per cell tupled with DataFrame containing
            phenotype frequencies of counts per phenotype/total phenotypes for each cell
    """

    # Setup input and parameters
    if included_fovs is None:
        included_fovs = sorted(list(set(all_data[fov_col])))

    # Error Checking
    if not np.isin(included_fovs, all_data[fov_col]).all():
        raise ValueError("Points were not found in Expression Matrix")

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

    for i in range(len(included_fovs)):
        # Subsetting expression matrix to only include patients with correct fov label
        current_fov_idx = all_neighborhood_data.iloc[:, 0] == included_fovs[i]
        current_fov_neighborhood_data = all_neighborhood_data[current_fov_idx]

        # Get the subset of phenotypes included in the current fov
        fov_cluster_names = current_fov_neighborhood_data[cluster_name_col].drop_duplicates()

        # Retrieve point specific distance matrix from distance matrix dictionary
        dist_matrix = dist_matrices_dict[included_fovs[i]]

        # Get cell_neighbor_counts and cell_neighbor_freqs for points
        counts, freqs = spatial_analysis_utils.compute_neighbor_counts(
            current_fov_neighborhood_data, dist_matrix, distlim)

        # add to neighbor counts + freqs for only the matching phenos between fov and whole dataset
        cell_neighbor_counts.loc[current_fov_neighborhood_data.index, fov_cluster_names] = counts
        cell_neighbor_freqs.loc[current_fov_neighborhood_data.index, fov_cluster_names] = freqs

    return cell_neighbor_counts, cell_neighbor_freqs
