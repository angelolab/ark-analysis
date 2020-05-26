import pandas as pd
import xarray as xr
import numpy as np
import scipy
import statsmodels
from statsmodels.stats.multitest import multipletests

# Erin's Data Inputs

# cell_array = pd.read_csv("/Users/jaiveersingh/Downloads/SpatialEn"
#                          "richment/granA_cellpheno_CS-asinh-norm_revised.csv")
# marker_thresholds = pd.read_csv("/Users/jaiveersingh/Downloads/Sp"
#                                 "atialEnrichment/markerThresholds.csv")
# dist_matrix = np.asarray(pd.read_csv("/Users/jaiveersingh/Documen"
#                                      "ts/MATLAB/distancesMat5.csv",
#                                      header=None))


def compute_close_cell_num(patient_data_markers, thresh_vec,
                           dist_mat, marker_num, dist_lim):
    """Finds positive cell labels and creates matrix with counts for cells positive for corresponding markers.

    This function loops through all the included markers in the patient data and identifies cell labels positive for
    corresponding markers. It then subsets the distance matrix to only include these positive cells and records
    interactions based on whether cells are close to each other (within the dist_lim). It then stores the number of
    interactions in the index of close_num corresponding to both markers (for instance markers 1 and 2 would be in
    index [0, 1]).

    Args:
        patient_data_markers: cell expression data of markers
            for the specific point
        thresh_vec: matrix of thresholds column for markers
        dist_mat: cells x cells matrix with the euclidian
            distance between centers of corresponding cells
        marker_num: number of markers in expresion data
        dist_lim: threshold for spatial enrichment distance proximity

    Returns:
        close_num: marker x marker matrix with counts for cells
            positive for corresponding markers
        marker1_num: list of number of cell labels for marker 1
        marker2_num: list of number of cell labels for marker 2"""

    # Create close_num, marker1_num, and marker2_num
    close_num = np.zeros((marker_num, marker_num), dtype='int')
    marker1_num = []
    marker2_num = []

    for j in range(0, marker_num):
        # Identify cell labels that are positive for respective markers
        marker1_thresh = thresh_vec.iloc[j]
        marker1posinds = patient_data_markers[patient_data_markers.columns[j]] > marker1_thresh
        marker1_num.append(sum(marker1posinds))
        for k in range(0, marker_num):
            # Identify cell labels that are positive for the kth marker
            marker2_thresh = thresh_vec.iloc[k]
            marker2posinds = patient_data_markers[patient_data_markers.columns[k]] > marker2_thresh
            marker2_num.append(sum(marker2posinds))
            # Subset the distance matrix to only include cells positive for both markers j and k
            trunc_dist_mat = dist_mat[np.ix_(np.asarray(marker1posinds), np.asarray(marker2posinds))]
            # Binarize the truncated distance matrix to only include cells within distance limit
            trunc_dist_mat_bin = np.zeros(trunc_dist_mat.shape, dtype='int')
            trunc_dist_mat_bin[trunc_dist_mat < dist_lim] = 1
            # Record the number of interactions and store in close_num in index corresponding to both markers
            close_num[j, k] = np.sum(np.sum(trunc_dist_mat_bin))
    return close_num, marker1_num, marker2_num


def compute_close_cell_num_random(marker1_num, marker2_num,
                                  dist_mat, marker_num, dist_lim, bootstrap_num):
    """Uses bootstrapping to permute cell labels randomly and records the number of close cells (within the dit_lim)
    in that random setup.

    Args
        marker1_num: list of number of cell labels for marker 1
        marker2_num: list of number of cell labels for marker 2
        dist_mat: cells x cells matrix with the euclidian
            distance between centers of corresponding cells
        marker_num: number of markers in expresion data
        dist_lim: threshold for spatial enrichment distance proximity
        bootstrap_num: number of permutations

    Returns
        close_num_rand: random positive marker counts
            for every permutation in the bootstrap"""

    # Create close_num_rand
    close_num_rand = np.zeros((
        marker_num, marker_num, bootstrap_num), dtype='int')

    for j in range(0, marker_num):
        for k in range(0, marker_num):
            for r in range(0, bootstrap_num):
                # Select same amount of random cell labels as positive ones in same marker in close_num
                marker1_labels_rand = np.random.choice(a=range(dist_mat.shape[0]), size=marker1_num[j], replace=True)
                marker2_labels_rand = np.random.choice(a=range(dist_mat.shape[0]), size=marker2_num[k], replace=True)
                # Subset the distance matrix to only include positive randomly selected cell labels
                rand_trunc_dist_mat = dist_mat[np.ix_(np.asarray(
                    marker1_labels_rand), np.asarray(marker2_labels_rand))]
                # Binarize the truncated distance matrix to only include cells within distance limit
                rand_trunc_dist_mat_bin = np.zeros(rand_trunc_dist_mat.shape, dtype='int')
                rand_trunc_dist_mat_bin[rand_trunc_dist_mat < dist_lim] = 1
                # Record the number of interactions and store in close_num_rand in the index
                # corresponding to both markers, for every permutation
                close_num_rand[j, k, r] = np.sum(np.sum(rand_trunc_dist_mat_bin))
    return close_num_rand


def calculate_enrichment_stats(close_num, close_num_rand):
    """Calculates z score and p values from spatial enrichment analysis.

    Args:
        close_num: marker x marker matrix with counts for cells
            positive for corresponding markers
        close_num_rand: random positive marker counts
            for every permutation in the bootstrap

    Returns:
        z: z scores for corresponding markers
        muhat: predicted mean values of close_num_rand random distribution
        sigmahat: predicted standard deviation values of close_num_rand
            random distribution
        p: p values for corresponding markers, for both positive
            and negative enrichment
        h: matrix indicating whether
            corresponding marker interactions are significant
        adj_p: fdh_br adjusted p values"""
    # Get the number of markers and number of permutations
    marker_num = close_num.shape[0]
    bootstrap_num = close_num_rand.shape[2]

    # Create z, muhat, sigmahat, and p
    z = np.zeros((marker_num, marker_num))
    muhat = np.zeros((marker_num, marker_num))
    sigmahat = np.zeros((marker_num, marker_num))
    p_pos = np.zeros((marker_num, marker_num))
    p_neg = np.zeros((marker_num, marker_num))

    for j in range(0, marker_num):
        for k in range(0, marker_num):
            # Get close_num_rand value for every marker combination and reshape to use as input for norm fit
            tmp = np.reshape(close_num_rand[j, k, :], (bootstrap_num, 1))
            # Get muhat and sigmahat values for distribution from 100 permutations
            (muhat[j, k], sigmahat[j, k]) = scipy.stats.norm.fit(tmp)
            # Calculate z score based on distribution
            z[j, k] = (close_num[j, k] - muhat[j, k]) / sigmahat[j, k]
            # Calculate both positive and negative enrichment p values
            p_pos[j, k] = (1 + (np.sum(tmp >= close_num[j, k]))) / (bootstrap_num + 1)
            p_neg[j, k] = (1 + (np.sum(tmp <= close_num[j, k]))) / (bootstrap_num + 1)

    # Get fdh_br adjusted p values
    p_summary = np.zeros_like(p_pos[:, :])
    for j in range(0, marker_num):
        for k in range(0, marker_num):
            # Use negative enrichment p values if the z score is negative, and vice versa
            if z[j, k] > 0:
                p_summary[j, k] = p_pos[j, k]
            else:
                p_summary[j, k] = p_neg[j, k]
    (h, adj_p, aS, aB) = statsmodels.stats.multitest.multipletests(
        p_summary, alpha=.05)

    # Create an Xarray with the dimensions (stats variables, number of markers, number of markers)
    stats_data = np.stack((z, muhat, sigmahat, p_pos, p_neg, h, adj_p), axis=0)
    coords = [["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"],
              range(stats_data[0].data.shape[0]), range(stats_data[0].data.shape[1])]
    dims = ["stats", "rows", "cols"]
    stats_xr = xr.DataArray(stats_data, coords=coords, dims=dims)
    # return z, muhat, sigmahat, p_pos, p_neg, h, adj_p
    return stats_xr


def calculate_channel_spatial_enrichment(dist_matrix, marker_thresholds, all_patient_data,
                                         excluded_colnames=None, points=None,
                                         patient_idx=30, dist_lim=100, bootstrap_num=1000):
    """Spatial enrichment analysis to find significant interactions between cells expressing different markers.
    Uses bootstrapping to permute cell labels randomly.

    Args:
        dist_matrix: cells x cells matrix with the euclidian
            distance between centers of corresponding cells
        marker_thresholds: threshold values for positive marker expression
        all_patient_data: data including points, cell labels, and
            cell expression matrix for all markers
        excluded_colnames: all column names that are not markers. If argument is none, default is
            ["cell_size", "Background", "HH3",
            "summed_channel", "label", "area",
            "eccentricity", "major_axis_length", "minor_axis_length",
            "perimeter", "fov"]
        points: patient labels to include in analysis. If argument is none, default is all labels used.
        patient_idx: columns with patient labels. Default is 30.
        dist_lim: cell proximity threshold. Default is 100.
        cell_label_idx: column with cell labels. Default is 24.
        bootstrap_num: number of permutations for bootstrap. Default is 1000.

    Returns:
        values: a list with each element consisting of a tuple of
            closenum and closenumrand for each point included in the analysis
        stats: an Xarray with dimensions (points, stats, number of markers, number of markers) The included stats
            variables are:
            z, muhat, sigmahat, p, h, adj_p, and marker_titles for each point in the analysis"""

    # Setup input and parameters
    num_points = 0
    if points is None:
        points = list(set(all_patient_data.iloc[:, patient_idx]))
        num_points = len(points)
    values = []
    # stats = []

    if excluded_colnames is None:
        excluded_colnames = ["cell_size", "Background", "HH3",
                             "summed_channel", "label", "area",
                             "eccentricity", "major_axis_length", "minor_axis_length",
                             "perimeter", "fov"]

    # Error Checking
    if not np.isin(excluded_colnames, all_patient_data.columns).all():
        raise ValueError("Column names were not found in Expression Matrix")

    if not np.isin(points, all_patient_data.iloc[:, patient_idx]).all():
        raise ValueError("Points were not found in Expression Matrix")

    # Subsets the expression matrix to only have marker columns
    data_markers = all_patient_data.drop(excluded_colnames, axis=1)
    # List of all markers
    marker_titles = data_markers.columns
    # Length of marker list
    marker_num = len(marker_titles)

    # Create stats Xarray with the dimensions (points, stats variables, number of markers, number of markers)
    stats_raw_data = np.zeros((num_points, 7, marker_num, marker_num))
    coords = [points, ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"], marker_titles,
              marker_titles]
    dims = ["points", "stats", "marker1", "marker2"]
    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

    # Subsetting threshold matrix to only include column with threshold values
    thresh_vec = marker_thresholds.iloc[:, 1]

    for point in points:
        # Subsetting expression matrix to only include patients with correct label
        patient_ids = all_patient_data.iloc[:, patient_idx] == point
        patient_data = all_patient_data[patient_ids]
        # Patients with correct label, and only columns of markers
        patient_data_markers = data_markers[patient_ids]

        # Get close_num and close_num_rand
        close_num, marker1_num, marker2_num = compute_close_cell_num(
            patient_data_markers, thresh_vec, dist_matrix, marker_num, dist_lim)
        close_num_rand = compute_close_cell_num_random(
            marker1_num, marker2_num, dist_matrix, marker_num, dist_lim, bootstrap_num)
        values.append((close_num, close_num_rand))
        # Get z, p, adj_p, muhat, sigmahat, and h
        # z, muhat, sigmahat, p, h, adj_p = calculate_enrichment_stats(close_num, close_num_rand)
        stats_xr = calculate_enrichment_stats(close_num, close_num_rand)
        # stats.append((z, muhat, sigmahat, p, h, adj_p, marker_titles))
        stats.loc[point, :, :] = stats_xr.values
    return values, stats
