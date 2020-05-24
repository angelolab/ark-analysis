import pandas as pd
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


# Test array inputs

# pv_cellarray = pd.read_csv("/Users/jaiveersingh/De
# sktop/tests/pvcelllabel.csv")
#
# pv_distmat = np.asarray(pd.read_csv("/Users/jaiveersingh/
# Desktop/tests/pvdistmat.csv", header = None))
#
# pv_cellarrayn = pd.read_csv("/Users/jaiv
# eersingh/Desktop/tests/pvcellarrayN.csv")
#
# pv_distmatn = np.asarray(pd.read_csv("/Users/jaiveers
# ingh/Desktop/tests/pvdistmatN.csv", header = None))
#
# randMat = np.random.randint(0,200,size=(60,60))
# np.fill_diagonal(randMat, 0)
#
# pv_cellarrayr = pd.read_csv("/Users/jaiveers"
# ingh/Desktop/tests/pvcellarrayR.csv")


def helper_function_closenum(patient_data, patient_data_markers, thresh_vec,
                             dist_mat, marker_num, dist_lim, cell_label_idx):
    """Finds positive cell labels and creates matrix with counts for cells positive for corresponding markers.

    Args:
        patient_data: cell expression data for the specific point
        patient_data_markers: cell expression data of markers
            for the specific point
        thresh_vec: matrix of thresholds column for markers
        dist_mat: cells x cells matrix with the euclidian
            distance between centers of corresponding cells
        marker_num: number of markers in expresion data
        dist_lim: threshold for spatial enrichment distance proximity
        cell_label_idx: column with cell labels

    Returns:
        close_num: marker x marker matrix with counts for cells
            positive for corresponding markers
        marker1_num: list of number of cell labels for marker 1
        marker2_num: list of number of cell labels for marker 2"""

    # create close_num, marker1_num, and marker2_num
    close_num = np.zeros((marker_num, marker_num), dtype='int')
    marker1_num = []
    marker2_num = []

    # later implement outside for loop for dif points
    for j in range(0, marker_num):
        # identify cell labels that are positive for respective markers
        marker1_thresh = thresh_vec.iloc[j]
        marker1posinds = patient_data_markers[patient_data_markers.columns[j]] > marker1_thresh
        marker1poslabels = patient_data.loc[marker1posinds, patient_data.columns[cell_label_idx]]
        marker1_num.append(len(marker1poslabels))
        for k in range(0, marker_num):
            # identify cell labels that are positive for above marker and all other markers
            marker2_thresh = thresh_vec.iloc[k]
            marker2posinds = patient_data_markers[patient_data_markers.columns[k]] > marker2_thresh
            marker2poslabels = patient_data.loc[marker2posinds, patient_data.columns[cell_label_idx]]
            marker2_num.append(len(marker2poslabels))
            # subset the distance matrix to only include positive cell labels
            trunc_dist_mat = dist_mat[np.ix_(
                np.asarray(marker1poslabels - 1), np.asarray(
                    marker2poslabels - 1))]
            # binarize the truncated distance matrix to only include cells within distance limit
            trunc_dist_mat_bin = np.zeros(trunc_dist_mat.shape, dtype='int')
            trunc_dist_mat_bin[trunc_dist_mat < dist_lim] = 1
            # record the number of interactions and store in close_num in index corresponding to both markers
            close_num[j, k] = np.sum(np.sum(trunc_dist_mat_bin))
    return close_num, marker1_num, marker2_num


def helper_function_closenumrand(marker1_num, marker2_num, patient_data,
                                 dist_mat, marker_num, dist_lim, cell_label_idx, bootstrap_num):
    """Uses bootstrapping to permute cell labels randomly.

    Args
        marker1_num: list of number of cell labels for marker 1
        marker2_num: list of number of cell labels for marker 2
        patient_data: cell expression data for the specific point
        dist_mat: cells x cells matrix with the euclidian
            distance between centers of corresponding cells
        marker_num: number of markers in expresion data
        dist_lim: threshold for spatial enrichment distance proximity
        cell_label_idx: column with cell labels
        bootstrap_num: number of permutations

    Returns
        close_num_rand: random positive marker counts
            for every permutation in the bootstrap"""

    # create close_num_rand
    close_num_rand = np.zeros((
        marker_num, marker_num, bootstrap_num), dtype='int')

    for j in range(0, marker_num):
        for k in range(0, marker_num):
            for r in range(0, bootstrap_num):
                # select same amount of random cell labels as positive ones in same marker in close_num
                marker1labelsrand = patient_data[
                    patient_data.columns[cell_label_idx]].sample(
                        n=marker1_num[j], replace=True)
                marker2labelsrand = patient_data[
                    patient_data.columns[cell_label_idx]].sample(
                        n=marker2_num[k], replace=True)
                # subset the distance matrix to only include positive randomly selected cell labels
                rand_trunc_dist_mat = dist_mat[np.ix_(
                    np.asarray(marker1labelsrand - 1), np.asarray(
                        marker2labelsrand - 1))]
                # binarize the truncated distance matrix to only include cells within distance limit
                rand_trunc_dist_mat_bin = np.zeros(rand_trunc_dist_mat.shape, dtype='int')
                rand_trunc_dist_mat_bin[rand_trunc_dist_mat < dist_lim] = 1
                # record the number of interactions and store in close_num_rand in index \
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
    # get the number of markers and number of permutations
    marker_num = close_num.shape[0]
    bootstrap_num = close_num_rand.shape[2]

    # create z, muhat, sigmahat, and p
    z = np.zeros((marker_num, marker_num))
    muhat = np.zeros((marker_num, marker_num))
    sigmahat = np.zeros((marker_num, marker_num))
    p = np.zeros((marker_num, marker_num, 2))

    for j in range(0, marker_num):
        for k in range(0, marker_num):
            # get close_num_rand value for every marker combination and reshape to use as input for norm fit
            tmp = np.reshape(close_num_rand[j, k, :], (bootstrap_num, 1))
            # get muhat and sigmahat values for distribution from 100 permutations
            (muhat[j, k], sigmahat[j, k]) = scipy.stats.norm.fit(tmp)
            # calculate z score based on distribution
            z[j, k] = (close_num[j, k] - muhat[j, k]) / sigmahat[j, k]
            # calculate both positive and negative enrichment p values
            p[j, k, 0] = (1 + (np.sum(tmp >= close_num[j, k]))) / (bootstrap_num + 1)
            p[j, k, 1] = (1 + (np.sum(tmp <= close_num[j, k]))) / (bootstrap_num + 1)

    # get fdh_br adjusted p values
    p_summary = np.zeros_like(p[:, :, 0])
    for j in range(0, marker_num):
        for k in range(0, marker_num):
            # use negative enrichment p values if the z score is negative, and vice versa
            if z[j, k] > 0:
                p_summary[j, k] = p[j, k, 0]
            else:
                p_summary[j, k] = p[j, k, 1]
    (h, adj_p, aS, aB) = statsmodels.stats.multitest.multipletests(
        p_summary, alpha=.05)

    return z, muhat, sigmahat, p, h, adj_p


def calculate_channel_spatial_enrichment(dist_matrix, marker_thresholds, cell_array,
                                         excluded_colnames=None, points=None,
                                         patient_idx=30, dist_lim=100, cell_label_idx=24, bootstrap_num=100):
    """Spatial enrichment analysis to find significant interactions between cells expressing different markers.
    Uses bootstrapping to permute cell labels randomly.

    Args:
        dist_matrix: cells x cells matrix with the euclidian
            distance between centers of corresponding cells
        marker_thresholds: threshold values for positive marker expression
        cell_array: data including points, cell labels, and
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
        bootstrap_num: number of permutations for bootstrap. Default is 100.

    Returns:
        values: a list with each element consisting of a tuple of
            closenum and closenumrand for each point included in the analysis
        stats: a list with each element consisting of a tuple of
            z, muhat, sigmahat, p, h, adj_p, and marker_titles for each point in the analysis"""

    # Setup input and parameters
    if points is None:
        points = list(set(cell_array.iloc[:, patient_idx]))
    values = []
    stats = []

    if excluded_colnames is None:
        excluded_colnames = ["cell_size", "Background", "HH3",
                             "summed_channel", "label", "area",
                             "eccentricity", "major_axis_length", "minor_axis_length",
                             "perimeter", "fov"]

    # Error Checking
    if not np.isin(excluded_colnames, cell_array.columns).all():
        raise ValueError("Column names were not found in Expression Matrix")

    if not np.isin(points, cell_array.iloc[:, patient_idx]).all():
        raise ValueError("Points were not found in Expression Matrix")

    # subsets the expression matrix to only have marker columns
    data_markers = cell_array.drop(excluded_colnames, axis=1)
    # list of all markers
    marker_titles = data_markers.columns
    # length of marker list
    marker_num = len(marker_titles)

    # subsetting threshold matrix to only include column with threshold values
    thresh_vec = marker_thresholds.iloc[:, 1]

    for i in points:
        # subsetting expression matrix to only include patients with correct label
        patient_ids = cell_array.iloc[:, patient_idx] == i
        patient_data = cell_array[patient_ids]
        # patients with correct label, and only columns of markers
        patient_data_markers = data_markers[patient_ids]

        # get close_num and close_num_rand
        close_num, marker1_num, marker2_num = helper_function_closenum(
            patient_data, patient_data_markers, thresh_vec, dist_matrix, marker_num, dist_lim, cell_label_idx)
        close_num_rand = helper_function_closenumrand(
            marker1_num, marker2_num, patient_data, dist_matrix, marker_num, dist_lim, cell_label_idx, bootstrap_num)
        values.append((close_num, close_num_rand))
        # get z, p, adj_p, muhat, sigmahat, and h
        z, muhat, sigmahat, p, h, adj_p = calculate_enrichment_stats(close_num, close_num_rand)
        stats.append((z, muhat, sigmahat, p, h, adj_p, marker_titles))

    return values, stats
