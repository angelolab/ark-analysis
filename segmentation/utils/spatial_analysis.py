import pandas as pd
import xarray as xr
import numpy as np
from segmentation.utils import spatial_analysis_utils
import importlib
importlib.reload(spatial_analysis_utils)


def calculate_channel_spatial_enrichment(dist_matrices, marker_thresholds, all_data,
                                         excluded_colnames=None, fovs=None,
                                         dist_lim=100, bootstrap_num=1000):
    """Spatial enrichment analysis to find significant interactions between cells expressing different markers.
    Uses bootstrapping to permute cell labels randomly.

    Args:
        dist_matrices: A dictionary that contains a cells x cells matrix with the euclidian
            distance between centers of corresponding cells for every fov
        marker_thresholds: threshold values for positive marker expression
        all_data: data including points, cell labels, and
            cell expression matrix for all markers
        excluded_colnames: all column names that are not markers. If argument is none, default is
            ["cell_size", "Background", "HH3",
            "summed_channel", "label", "area",
            "eccentricity", "major_axis_length", "minor_axis_length",
            "perimeter", "fov"]
        fovs: patient labels to include in analysis. If argument is none, default is all labels used.
        dist_lim: cell proximity threshold. Default is 100.
        bootstrap_num: number of permutations for bootstrap. Default is 1000.
        seed: the value to set for randomized seed. Useful for testing. Default None.

    Returns:
        values: a list with each element consisting of a tuple of
            closenum and closenumrand for each point included in the analysis
        stats: an Xarray with dimensions (points, stats, number of markers, number of markers) The included stats
            variables are:
            z, muhat, sigmahat, p, h, adj_p, and marker_titles for each point in the analysis"""

    fov_col = "SampleID"

    # Setup input and parameters
    if fovs is None:
        fovs = list(set(all_data[fov_col]))
        num_fovs = len(fovs)
    else:
        num_fovs = len(fovs)

    values = []

    if excluded_colnames is None:
        excluded_colnames = ["cell_size", "Background", "HH3",
                             "summed_channel", "label", "area",
                             "eccentricity", "major_axis_length", "minor_axis_length",
                             "perimeter", "fov"]

    # Error Checking
    if not np.isin(excluded_colnames, all_data.columns).all():
        raise ValueError("Column names were not found in Expression Matrix")

    if not np.isin(fovs, all_data[fov_col]).all():
        raise ValueError("Points were not found in Expression Matrix")

    # Subsets the expression matrix to only have marker columns
    data_markers = all_data.drop(excluded_colnames, axis=1)
    # List of all markers
    marker_titles = data_markers.columns
    # Length of marker list
    marker_num = len(marker_titles)

    # Check to see if order of marker thresholds is same as in expression matrix
    if not (list(marker_thresholds.iloc[:, 0]) == marker_titles).any():
        raise ValueError("Threshold Markers do not match markers in Expression Matrix")

    # Create stats Xarray with the dimensions (points, stats variables, number of markers, number of markers)
    stats_raw_data = np.zeros((num_fovs, 7, marker_num, marker_num))
    coords = [fovs, ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"], marker_titles,
              marker_titles]
    dims = ["points", "stats", "marker1", "marker2"]
    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    # Subsetting threshold matrix to only include column with threshold values
    thresh_vec = marker_thresholds.iloc[:, 1]

    for i in range(0, len(fovs)):
        # Subsetting expression matrix to only include patients with correct label
        patient_ids = all_data[fov_col] == fovs[i]
        fov_data = all_data[patient_ids]
        # Patients with correct label, and only columns of markers
        fov_channel_data = data_markers[patient_ids]

        # Subset the distance matrix dictionary to only include the distance matrix for the correct point
        dist_matrix = dist_matrices[str(fovs[i])]

        # Get close_num and close_num_rand
        close_num, marker1_num, marker2_num = spatial_analysis_utils.compute_close_cell_num(
            dist_mat=dist_matrix, dist_lim=100, num=marker_num, analysis_type="Channel",
            fov_data=fov_data, fov_channel_data=fov_channel_data, thresh_vec=thresh_vec)

        close_num_rand = spatial_analysis_utils.compute_close_cell_num_random(
            marker1_num, marker2_num, dist_matrix, marker_num, dist_lim, bootstrap_num)
        values.append((close_num, close_num_rand))

        # Get z, p, adj_p, muhat, sigmahat, and h
        stats_xr = spatial_analysis_utils.calculate_enrichment_stats(close_num, close_num_rand)
        stats.loc[fovs[i], :, :] = stats_xr.values
    return values, stats


def calculate_cluster_spatial_enrichment(all_data, dist_mats, fovs=None,
                                         bootstrap_num=1000, dist_lim=100):
    """Spatial enrichment analysis based on cell phenotypes to find significant interactions between different
    cell types, looking for both positive and negative enrichment. Uses bootstrapping to permute cell labels randomly.

    Args:
        all_data: data including points, cell labels, and
            cell expression matrix for all markers
        dist_mats: A dictionary that contains a cells x cells matrix with the euclidian
            distance between centers of corresponding cells for every fov
        fovs: patient labels to include in analysis. If argument is none, default is all labels used
        bootstrap_num: number of permutations for bootstrap. Default is 1000
        dist_lim: cell proximity threshold. Default is 100

    Returns:
        values: a list with each element consisting of a tuple of
            closenum and closenumrand for each point included in the analysis
        stats: an Xarray with dimensions (points, stats, number of markers, number of markers) The included stats
            variables are:
            z, muhat, sigmahat, p, h, adj_p, and marker_titles for each point in the analysis"""

    fov_col = "SampleID"
    cell_type_col = "cell_type"
    flowsom_col = "FlowSOM_ID"
    cell_label_col = "cellLabelInImage"

    # Setup input and parameters
    if fovs is None:
        fovs = list(set(all_data[fov_col]))
        num_fovs = len(fovs)
    else:
        num_fovs = len(fovs)

    values = []

    # Error Checking
    if not np.isin(fovs, all_data[fov_col]).all():
        raise ValueError("Points were not found in Expression Matrix")

    # Extract the names of the cell phenotypes
    pheno_titles = all_data[cell_type_col].drop_duplicates()
    # Extract the columns with the cell phenotype codes
    pheno_codes = all_data[flowsom_col].drop_duplicates()
    # Get the total number of phenotypes
    pheno_num = len(pheno_codes)

    # Subset matrix to only include the columns with the patient label, cell label, and cell phenotype
    fov_cluster_data = all_data[[fov_col, cell_label_col, flowsom_col]]

    # Create stats Xarray with the dimensions (points, stats variables, number of markers, number of markers)
    stats_raw_data = np.zeros((num_fovs, 7, pheno_num, pheno_num))
    coords = [fovs, ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"], pheno_titles, pheno_titles]
    dims = ["points", "stats", "pheno1", "pheno2"]
    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    for i in range(0, len(fovs)):
        # Subsetting expression matrix to only include patients with correct label
        patient_ids = fov_cluster_data.iloc[:, 0] == fovs[i]
        fov_data = fov_cluster_data[patient_ids]

        # Subset the distance matrix dictionary to only include the distance matrix for the correct point
        dist_mat = dist_mats[str(fovs[i])]

        # Get close_num and close_num_rand
        close_num, pheno1_num, pheno2_num = spatial_analysis_utils.compute_close_cell_num(
            dist_mat=dist_mat, dist_lim=dist_lim, num=pheno_num, analysis_type="Cluster",
            fov_data=fov_data, pheno_codes=pheno_codes)
        close_num_rand = spatial_analysis_utils.compute_close_cell_num_random(
            pheno1_num, pheno2_num, dist_mat, pheno_num, dist_lim, bootstrap_num)
        values.append((close_num, close_num_rand))

        # Get z, p, adj_p, muhat, sigmahat, and h
        stats_xr = spatial_analysis_utils.calculate_enrichment_stats(close_num, close_num_rand)
        stats.loc[fovs[i], :, :] = stats_xr.values
    return values, stats


def create_neighborhood_matrix(all_data, dist_matrices, fov_list=None, distlim=50, fov_col="SampleID",
                               flowsom_col="FlowSOM_ID", cell_label_col="cellLabelInImage", cell_type_col="cell_type"):
    """Calculates the number of neighbor phenotypes for each cell. The function counts each cell as one of its own
    neighbors in this implementation.

        Args:
            all_data: data for the all fovs in the form of a pandas DF, including the columns of SampleID (fovs),
                cellLabelInImage (the cell label), and FlowSOM_ID (the cell phenotype id).
            dist_matrices: A dictionary that contains a cells x cells matrix with the euclidian
                distance between centers of corresponding cells for every fov
            fov_list: patient labels to include in analysis. If argument is none, default is all labels used.
            distlim: cell proximity threshold. Default is 50.
            fov_col: column with the cell fovs (Default is SampleID)
            flowsom_col: column with the cell phenotype IDs (Default is FlowSOM_ID)
            cell_label_col: column with the cell labels (Default is cellLabelInImage)
            cell_type_col: column with the cell types (Default is cell_type)
        Returns:
            cell_neighbor_counts: matrix with phenotype counts per cell
            cell_neighbor_freqs: matrix with phenotype frequencies of
                counts per phenotype/total phenotypes for each cell
            cell_count: current cell in analysis"""

    # Setup input and parameters
    if fov_list is None:
        fov_list = sorted(list(set(all_data[fov_col])))

    # Error Checking
    if not np.isin(fov_list, all_data[fov_col]).all():
        raise ValueError("Points were not found in Expression Matrix")

    # Get the phenotypes
    pheno_titles = all_data[cell_type_col].drop_duplicates()

    # Subset just the sampleID, cellLabelInImage, and FlowSOMID
    all_data = all_data[[fov_col, cell_label_col, flowsom_col, cell_type_col]]
    # Extract the columns with the cell phenotype codes
    pheno_codes = all_data[flowsom_col].drop_duplicates()
    # Get the total number of phenotypes
    pheno_num = len(pheno_codes)

    # initiate empty matrices for cell neighborhood data
    cell_neighbor_counts = pd.DataFrame(np.zeros((all_data.shape[0], pheno_num + 2)))

    # Replace the first and second columns of cell_neighbor_counts with the fovs and cell labels respectively
    cell_neighbor_counts[[0, 1]] = all_data[[fov_col, cell_label_col]]

    # Rename the columns to match cell phenotypes
    cols = [fov_col, cell_label_col] + list(pheno_titles)
    cell_neighbor_counts.columns = cols

    cell_neighbor_freqs = cell_neighbor_counts.copy(deep=True)

    for i in range(len(fov_list)):
        # Subsetting expression matrix to only include patients with correct label
        patient_idx = all_data.iloc[:, 0] == fov_list[i]
        fov_data = all_data[patient_idx]

        # Get the subset of phenotypes included in the current fov
        fov_pheno_titles = fov_data[cell_type_col].drop_duplicates()

        # Subset the distance matrix dictionary to only include the distance matrix for the correct point
        dist_matrix = dist_matrices[str(fov_list[i])]

        # Get cell_neighbor_counts and cell_neighbor_freqs for points
        counts, freqs = spatial_analysis_utils.compute_neighbor_counts(
            fov_data, dist_matrix, distlim)

        # add to neighbor counts + freqs for only the matching phenotypes between the fov and the whole dataset
        cell_neighbor_counts.loc[fov_data.index, fov_pheno_titles] = counts
        cell_neighbor_freqs.loc[fov_data.index, fov_pheno_titles] = freqs

    return cell_neighbor_counts, cell_neighbor_freqs
