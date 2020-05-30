import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import statsmodels
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import cdist


def calc_dist_matrix(label_map):
    """Generate matrix of distances between center of pairs of cells

    Args:
        label_map: numpy array with unique cells given unique pixel labels
    Returns:
        dist_matrix: cells x cells matrix with the euclidian
        distance between centers of corresponding cells"""
    dist_mats_list = []
    for i in range(0, label_map.shape[0]):
        props = skimage.measure.regionprops(label_map[i, :, :])
        a = []
        for j in range(len(props)):
            a.append(props[j].centroid)
        centroids = np.array(a)
        dist_matrix = cdist(centroids, centroids)
        dist_mats_list.append(dist_matrix)
    dist_mats = np.stack(dist_mats_list, axis=0)
    # label_map.coords["fovs"]
    coords = [range(len(dist_mats)), range(dist_mats[0].data.shape[0]), range(dist_mats[0].data.shape[1])]
    dims = ["points", "rows", "cols"]
    dist_mats_xr = xr.DataArray(dist_mats, coords=coords, dims=dims)
    return dist_mats_xr


def compute_close_cell_num(dist_mat, dist_lim, num, analysis_type, cell_label_idx=None,
                         patient_data=None, cell_pheno_idx=None, pheno_codes=None, patient_data_markers=None,
                         label_idx=None, thresh_vec=None):
    """Finds positive cell labels and creates matrix with counts for cells positive for corresponding markers.

    This function loops through all the included markers in the patient data and identifies cell labels positive for
    corresponding markers. It then subsets the distance matrix to only include these positive cells and records
    interactions based on whether cells are close to each other (within the dist_lim). It then stores the number of
    interactions in the index of close_num corresponding to both markers (for instance markers 1 and 2 would be in
    index [0, 1]).

    Args:
        patient_data_markers: cell expression data of markers
            for the specific point
        label_idx: column of cell labels
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
    close_num = np.zeros((num, num), dtype='int')
    mark1_num = []
    mark2_num = []

    for j in range(0, num):
        # Identify cell labels that are positive for respective markers, based on type of analysis
        mark1poslabels = None
        if analysis_type == "Cell Label":
            pheno1 = pheno_codes.iloc[j]
            pheno1posinds = patient_data.iloc[:, cell_pheno_idx] == pheno1
            mark1poslabels = patient_data.iloc[:, cell_label_idx][pheno1posinds]
        else:
            marker1_thresh = thresh_vec.iloc[j]
            marker1posinds = patient_data_markers[patient_data_markers.columns[j]] > marker1_thresh
            mark1poslabels = label_idx[marker1posinds]
        mark1_num.append(len(mark1poslabels))
        for k in range(0, num):
            mark2poslabels = None
            if analysis_type == "Cell Label":
                pheno2 = pheno_codes.iloc[k]
                pheno2posinds = patient_data.iloc[:, cell_pheno_idx] == pheno2
                mark2poslabels = patient_data.iloc[:, cell_label_idx][pheno2posinds]
            else:
                marker2_thresh = thresh_vec.iloc[k]
                marker2posinds = patient_data_markers[patient_data_markers.columns[k]] > marker2_thresh
                mark2poslabels = label_idx[marker2posinds]
            mark2_num.append(len(mark2poslabels))
            # Subset the distance matrix to only include cells positive for both markers j and k
            trunc_dist_mat = dist_mat[np.ix_(np.asarray(mark1poslabels - 1), np.asarray(mark2poslabels - 1))]
            # Binarize the truncated distance matrix to only include cells within distance limit
            trunc_dist_mat_bin = np.zeros(trunc_dist_mat.shape, dtype='int')
            trunc_dist_mat_bin[trunc_dist_mat < dist_lim] = 1
            close_num[j, k] = np.sum(np.sum(trunc_dist_mat_bin))
    return close_num, mark1_num, mark2_num


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
    return stats_xr
