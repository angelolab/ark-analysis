import pandas as pd
import numpy as np
import scipy
import statsmodels
from statsmodels.stats.multitest import multipletests

# Erin's Data Inputs

cell_array = pd.read_csv("/Users/jaiveersingh/Downloads/SpatialEn"
                         "richment/granA_cellpheno_CS-asinh-norm_revised.csv")
marker_thresholds = pd.read_csv("/Users/jaiveersingh/Downloads/Sp"
                                "atialEnrichment/markerThresholds.csv")
dist_matrix = np.asarray(pd.read_csv("/Users/jaiveersingh/Documen"
                                     "ts/MATLAB/distancesMat5.csv",
                                     header=None))


# Test array inputs
"""
pv_cellarray = pd.read_csv("/Users/jaiveersingh/De
sktop/tests/pvcelllabel.csv")

pv_distmat = np.asarray(pd.read_csv("/Users/jaiveersingh/
Desktop/tests/pvdistmat.csv", header = None))

pv_cellarrayn = pd.read_csv("/Users/jaiv
eersingh/Desktop/tests/pvcellarrayN.csv")

pv_distmatn = np.asarray(pd.read_csv("/Users/jaiveers
ingh/Desktop/tests/pvdistmatN.csv", header = None))

randMat = np.random.randint(0,200,size=(60,60))
np.fill_diagonal(randMat, 0)

pv_cellarrayr = pd.read_csv("/Users/jaiveers"
ingh/Desktop/tests/pvcellarrayR.csv")
"""


def spatial_analysis(dist_matrix, marker_thresholds, cell_array):
    """Spatial enrichment analysis to find significant interactions between cells
    expressing different markers.
    Uses bootstrapping to permute cell labels randomly.
            Args
                dist_matrix: cells x cells matrix with the euclidian
                    distance between centers of corresponding cells
                marker_thresholds: threshold
                    values for positive marker expression
                cell_array: data including points, cell labels, and
                    cell expression matrix for all markers
            Returns
                close_num: marker x marker matrix with counts for cells
                    positive for corresponding markers
                close_num_rand: random positive marker counts
                    for every permutation in the bootstrap
                z: z scores for corresponding markers
                muhat: predicted mean values
                    of close_num_rand random distribution
                sigmahat: predicted standard deviation values of close_num_rand
                    random distribution
                p: p values for corresponding markers, for both positive
                    and negative enrichment
                h: matrix indicating whether
                    corresponding marker interactions are significant
                adj_p: fdh_br adjusted p values
                marker_titles: list of markers"""

    # Setup input and parameters
    point = 6
    data_all = cell_array
    marker_inds = [7, 8] + list(range(10, 44))
    data_markers = data_all.loc[:, data_all.columns[marker_inds]]
    marker_titles = data_all.columns[marker_inds]
    marker_num = len(marker_titles)

    bootstrap_num = 100
    dist_lim = 100

    close_num = np.zeros((marker_num, marker_num), dtype='int')
    close_num_rand = np.zeros((
        marker_num, marker_num, bootstrap_num), dtype='int')

    patient_idx = 0
    cell_label_idx = 1

    marker_thresh = marker_thresholds
    thresh_vec = marker_thresh.iloc[1:38, 1]

    # Enrichment Analysis

    dist_mat = dist_matrix
    patient_ids = data_all.iloc[:, patient_idx] == point
    patient_data = data_all[patient_ids]
    patient_data_markers = data_markers[patient_ids]

    # later implement outside for loop for dif points
    for j in range(0, marker_num):
        marker1_thresh = thresh_vec.iloc[j]
        marker1posinds = patient_data_markers[patient_data_markers.columns[
            j]] > marker1_thresh
        marker1poslabels = patient_data.loc[
            marker1posinds, patient_data.columns[cell_label_idx]]
        marker1_num = len(marker1poslabels)
        for k in range(0, marker_num):
            marker2_thresh = thresh_vec.iloc[k]
            marker2posinds = patient_data_markers[patient_data_markers.columns[
                k]] > marker2_thresh
            marker2poslabels = patient_data.loc[
                marker2posinds, patient_data.columns[cell_label_idx]]
            marker2_num = len(marker2poslabels)
            trunc_dist_mat = dist_mat[np.ix_(
                np.asarray(marker1poslabels-1), np.asarray(
                    marker2poslabels-1))]

            trunc_dist_mat_bin = np.zeros(trunc_dist_mat.shape, dtype='int')
            trunc_dist_mat_bin[trunc_dist_mat < dist_lim] = 1

            close_num[j, k] = np.sum(np.sum(trunc_dist_mat_bin))
            # print(close_num[j,k])
            for r in range(0, bootstrap_num):
                marker1labelsrand = patient_data[
                    patient_data.columns[cell_label_idx]].sample(
                    n=marker1_num, replace=True)
                marker2labelsrand = patient_data[
                    patient_data.columns[cell_label_idx]].sample(
                    n=marker2_num, replace=True)
                rand_trunc_dist_mat = dist_mat[np.ix_(
                    np.asarray(marker1labelsrand - 1), np.asarray(
                        marker2labelsrand - 1))]
                rand_trunc_dist_mat_bin = np.zeros(
                    rand_trunc_dist_mat.shape, dtype='int')
                rand_trunc_dist_mat_bin[rand_trunc_dist_mat < dist_lim] = 1

                close_num_rand[j, k, r] = \
                    np.sum(np.sum(rand_trunc_dist_mat_bin))
    # z score, pval, and adj pval
    z = np.zeros((marker_num, marker_num))
    muhat = np.zeros((marker_num, marker_num))
    sigmahat = np.zeros((marker_num, marker_num))

    p = np.zeros((marker_num, marker_num, 2))

    for j in range(0, marker_num):
        for k in range(0, marker_num):
            tmp = np.reshape(close_num_rand[j, k, :], (bootstrap_num, 1))
            # print(tmp)
            (muhat[j, k], sigmahat[j, k]) = scipy.stats.norm.fit(tmp)
            # print(muhat[j, k])
            # print(sigmahat[j, k])
            z[j, k] = (close_num[j, k] - muhat[j, k]) / sigmahat[j, k]
            # print(z[j, k])
            p[j, k, 0] = (1 + (
                np.sum(tmp >= close_num[j, k]))) / (bootstrap_num + 1)
            # print(p[j, k, 0])
            p[j, k, 1] = (1 + (
                np.sum(tmp <= close_num[j, k]))) / (bootstrap_num + 1)
            # print(p[j, k, 1])

    p_summary = p[:, :, 0]
    for j in range(0, marker_num):
        for k in range(0, marker_num):
            if z[j, k] > 0:
                p_summary[j, k] = p[j, k, 0]
            else:
                p_summary[j, k] = p[j, k, 1]
    (h, adj_p, aS, aB) = statsmodels.stats.multitest.multipletests(
        p_summary, alpha=.05)

    return close_num, close_num_rand, z, \
        muhat, sigmahat, p, h, adj_p, marker_titles
    # end
