import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import statsmodels
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import cdist
import os


def calc_dist_matrix(label_map, ret=True, path=None):
    """Generate matrix of distances between center of pairs of cells

    Args:
        label_map: numpy array with unique cells given unique pixel labels
        ret: A boolean value indicating whether or not to return the dictionary file directly. Default is True.
        path: path to save file
    Returns:
        dist_matrix: A dictionary that contains a cells x cells matrix with the euclidian
            distance between centers of corresponding cells for every fov"""
    # Check that file path exists, if return is false
    if not ret:
        if not os.path.exists(path):
            raise ValueError("File path not valid")
    dist_mats_list = []
    # Extract list of fovs
    fovs = list(label_map.coords['fovs'].values)
    for i in range(0, label_map.shape[0]):
        props = skimage.measure.regionprops(label_map.loc[fovs[i], :, :, "segmentation_label"].values)
        a = []
        for j in range(len(props)):
            a.append(props[j].centroid)
        centroids = np.array(a)
        dist_matrix = cdist(centroids, centroids)
        dist_mats_list.append(dist_matrix)
    # Create dictionary to store distance matrices per fov
    dist_matrices = dict(zip([str(i) for i in fovs], dist_mats_list))
    # If ret is true, function will directly return the dictionary, else it will save it as a file
    if ret:
        return dist_matrices
    else:
        np.savez(path + "dist_matrices.npz", **dist_matrices)


def get_pos_cell_labels(analysis_type, pheno=None, fov_data=None,
                        thresh=None, fov_channel_data=None, cell_labels=None, col=None):
    """Based on the type of the analysis, finds positive labels that the current match phenotype or identifies cells
    with expression values for the current maker greater than the marker threshold.

    Args:
        analysis_type: type of analysis, either Cluster or Channel
        pheno: the current cell phenotype
        fov_data: data for the current patient
        thresh: current threshold for marker
        fov_channel_data: expression data for column markers for current patient
        cell_labels: the column of cell labels for current patient
        col: the current marker
    Returns:
        mark1poslabels: a list of all the positive labels"""

    if analysis_type == "Cluster":
        if pheno is None or fov_data is None:
            raise ValueError("Incorrect arguments passed for analysis type")
        # Subset only cells that are of the same phenotype
        pheno1posinds = fov_data["FlowSOM_ID"] == pheno
        # Get the cell labels of the cells of the phenotype
        mark1poslabels = fov_data.iloc[:, 1][pheno1posinds]
    else:
        if thresh is None or fov_channel_data is None or cell_labels is None or col is None:
            raise ValueError("Incorrect arguments passed for analysis type")
        # Subset only cells that are positive for the given marker
        marker1posinds = fov_channel_data[col] > thresh
        # Get the cell labels of the positive cells
        mark1poslabels = cell_labels[marker1posinds]
    return mark1poslabels


def compute_close_cell_num(dist_mat, dist_lim, num, analysis_type,
                           fov_data=None, fov_channel_data=None, pheno_codes=None,
                           thresh_vec=None):
    """Finds positive cell labels and creates matrix with counts for cells positive for corresponding markers.
    Computes close_num matrix for both Cell Label and Threshold spatial analyses.

    This function loops through all the included markers in the patient data and identifies cell labels positive for
    corresponding markers. It then subsets the distance matrix to only include these positive cells and records
    interactions based on whether cells are close to each other (within the dist_lim). It then stores the number of
    interactions in the index of close_num corresponding to both markers (for instance markers 1 and 2 would be in
    index [0, 1]).

    Args:
        dist_mat: cells x cells matrix with the euclidian
            distance between centers of corresponding cells
        dist_lim: threshold for spatial enrichment distance proximity
        num: number of markers or cell phenotypes, based on analysis
        analysis_type: type of analysis, either Cluster or Channel
        fov_data: data for specific patient in expression matrix
        fov_channel_data: data of only column markers for Channel Analysis
        pheno_codes: list all the cell phenotypes in Cluster Analysis
        thresh_vec: matrix of thresholds column for markers

    Returns:
        close_num: marker x marker matrix with counts for cells
            positive for corresponding markers
        marker1_num: list of number of cell labels for marker 1
        marker2_num: list of number of cell labels for marker 2"""
    # Initialize variables
    cell_labels = []

    # Assign column names for subsetting (cell labels)
    cell_label_col = "cellLabelInImage"

    # Subset data based on analysis type
    if analysis_type == "Channel":
        # Subsetting the column with the cell labels
        cell_labels = fov_data[cell_label_col]

    # Create close_num, marker1_num, and marker2_num
    close_num = np.zeros((num, num), dtype='int')
    mark1_num = []
    mark2_num = []

    for j in range(0, num):
        # Identify cell labels that are positive for respective markers or phenotypes, based on type of analysis
        if analysis_type == "Cluster":
            mark1poslabels = get_pos_cell_labels(analysis_type, pheno_codes.iloc[j], fov_data)
        else:
            mark1poslabels = get_pos_cell_labels(analysis_type, thresh=thresh_vec.iloc[j],
                                                 fov_channel_data=fov_channel_data, cell_labels=cell_labels,
                                                 col=fov_channel_data.columns[j])
        # Length of the number of positive cell labels
        mark1_num.append(len(mark1poslabels))
        for k in range(0, num):
            # Repeats what was done above for the same marker and all other markers in the analysis
            if analysis_type == "Cluster":
                mark2poslabels = get_pos_cell_labels(analysis_type, pheno_codes.iloc[k], fov_data)
            else:
                mark2poslabels = get_pos_cell_labels(analysis_type, thresh=thresh_vec.iloc[k],
                                                     fov_channel_data=fov_channel_data, cell_labels=cell_labels,
                                                     col=fov_channel_data.columns[k])
            mark2_num.append(len(mark2poslabels))
            # Subset the distance matrix to only include cells positive for both markers j and k
            trunc_dist_mat = dist_mat[np.ix_(np.asarray(mark1poslabels - 1, dtype='int'),
                                             np.asarray(mark2poslabels - 1, dtype='int'))]
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
                    marker1_labels_rand, dtype='int'), np.asarray(marker2_labels_rand, dtype='int'))]
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


def compute_neighbor_counts(fov_data, dist_matrix, distlim, pheno_num,
                            cell_neighbor_counts, cell_neighbor_freqs,
                            cell_label_col="cellLabelInImage", flowsom_col="FlowSOM_ID", cell_type_col="cell_type"):
    """Calculates the number of neighbor phenotypes for each cell

    Args:
        fov_data: data for the current fov
        dist_matrix: cells x cells matrix with the euclidian
            distance between centers of corresponding cells
        distlim: threshold for spatial enrichment distance proximity
        pheno_num: number of different cell phenotypes
        cell_neighbor_counts: matrix with phenotype counts per cell
        cell_neighbor_freqs: matrix with phenotype frequencies of
            counts per phenotype/total phenotypes for each cell
        cell_count: current cell in analysis
        cell_label_col: Column name with the cell labels
        flowsom_col: column name with the FlowSOM phenotype IDs
    Returns:
        cell_neighbor_counts: matrix with phenotype counts per cell
        cell_neighbor_freqs: matrix with phenotype frequencies of
            counts per phenotype/total phenotypes for each cell
        cell_count: current cell in analysis"""

    # for j in list(fov_data.index):
    #     # get specific cell
    #     cell = fov_data.loc[j, cell_label_col]
    #
    #     # Error checking to make sure correct cell labels are given (not negative)
    #     if not cell > 0:
    #         raise ValueError("Incorrect Cell Labels given as Input")
    #
    #     # get all cell neighbors within threshold distance
    #     cell_dist_mat = dist_matrix[int(cell - 1), :]
    #     cell_dist_mat_bin = np.zeros(cell_dist_mat.shape)
    #     # Binarize the cell_dist_mat, making all those within the distlim equal to 1
    #     cell_dist_mat_bin[cell_dist_mat < distlim] = 1
    #     cell_dist_mat_bin[cell_dist_mat == 0] = 0
    #
    #     # get indices (labels) of close cells
    #     neighbor_labels = np.argwhere(cell_dist_mat_bin > 0) + 1
    #     # filter out non-cellular objects (with no corresponding label in fov data)
    #     neighbor_labels_cells = np.intersect1d(neighbor_labels, fov_data[cell_label_col])
    #
    #     # count phenotypes in cell neighbors
    #     count_vec = np.zeros((1, pheno_num))
    #     # Only include the neighbor labels that are cell labels
    #     neighbor_inds = np.isin(fov_data[cell_label_col], neighbor_labels_cells)
    #     # Get the phenotypes of the neighbor labels by index
    #     pheno_vec = fov_data.loc[neighbor_inds, flowsom_col]
    #     count_vec[0, 0:(pheno_num - 1)] = [sum(pheno_vec == k) for k in range(1, pheno_num)]
    #     # add to neighborhood matrices
    #     cell_neighbor_counts.loc[cell_count, 2:] = count_vec[0, :]
    #     cell_neighbor_freqs.loc[cell_count, 2:] = (count_vec[0, :] / len(neighbor_labels_cells))
    #     # update cell counts
    #     cell_count += 1
    # return cell_count

    pheno_titles = fov_data[cell_type_col].drop_duplicates()

    # remove non-cell2cell distances
    cell_dist_mat = np.take(dist_matrix, fov_data[cell_label_col] - 1, 0)
    cell_dist_mat = np.take(cell_dist_mat, fov_data[cell_label_col] - 1, 1)

    # binarize distance matrix
    cell_dist_mat_bin = np.zeros(cell_dist_mat.shape)
    cell_dist_mat_bin[cell_dist_mat < distlim] = 1
    # cell_dist_mat_bin[cell_dist_mat == 0] = 0

    # get num_neighbors for freqs
    num_neighbors = np.sum(cell_dist_mat_bin, axis=0)

    # create the 'phenotype has cell?' matrix, excluding non cell-label rows (should match up w/ dist_matrix)
    # pheno_has_cell = pd.get_dummies(fov_data.iloc[fov_data[cell_label_col] - 1, 2]).to_numpy().T
    pheno_has_cell = pd.get_dummies(fov_data.iloc[:, 2]).to_numpy().T

    # dot binarized 'is neighbor?' matrix with pheno_has_cell to get counts
    counts = pheno_has_cell.dot(cell_dist_mat_bin).T

    # compute freqs with num_neighbors
    freqs = counts.T / num_neighbors

    # convert to pandas df and assign columns to phenos
    countsdf = pd.DataFrame(counts, columns=pheno_titles)
    freqsdf = pd.DataFrame(freqs.T, columns=pheno_titles)

    # add to neighbor counts + freqs for only the matching phenotypes between the fov and the whole dataset
    cell_neighbor_counts.loc[fov_data.index, countsdf.columns] = counts
    cell_neighbor_freqs.loc[fov_data.index, freqsdf.columns] = freqs.T
