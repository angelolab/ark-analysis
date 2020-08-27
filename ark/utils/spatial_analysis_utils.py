import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import statsmodels
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import cdist
import os


def calc_dist_matrix(label_map, centroid_labels=None, path=None):
    """Generate matrix of distances between center of pairs of cells

    Args:
        label_map (np array): array with unique cells given unique pixel labels
        centroid_labels (dict): the labels for each fov which are
            needed to add correct coordinates to access distance matrix
            needs to be of the same length and indexed corresponding to label_map.coords['fovs']
            if None, then default assume that cell labels are in order for each fov
        path (string): path to save file. If None, then will directly return
    Returns:
        dist_matrix (dict): contains a cells x cells matrix with the euclidian
            distance between centers of corresponding cells for every fov, note that each distance matrix
            is of type xarray"""
    # Check that file path exists, if given

    if path is not None:
        if not os.path.exists(path):
            raise ValueError("File path not valid")

    dist_mats_list = []

    # Extract list of fovs
    fovs = list(label_map.coords['fovs'].values)

    # generate centroid labels if None
    if centroid_labels is None:
        centroid_labels = {}

        for fov_val in fovs:
            fov_arr = label_map.loc[fov_val, :, :, 'segmentation_label'].values
            fov_arr_labels = np.unique(fov_arr[fov_arr > 1]).tolist()
            centroid_labels[fov_val] = fov_arr_labels

    for i in range(len(fovs)):
        # extract region properties of label map, then just get centroids
        props = skimage.measure.regionprops(label_map.loc[fovs[i], :, :, 'segmentation_label'].values)
        centroids = np.array([props[j].centroid for j in range(len(props))])

        # generate the distance matrix, then assign centroid_labels as coords
        dist_matrix = cdist(centroids, centroids)
        dist_mat_xarr = xr.DataArray(dist_matrix, coords=[centroid_labels[fovs[i]], centroid_labels[fovs[i]]])

        # append final result to dist_mats_list
        dist_mats_list.append(dist_mat_xarr)

    # Create dictionary to store distance matrices per fov
    dist_matrices = dict(zip([str(i) for i in fovs], dist_mats_list))

    # If ret is true, function will directly return the dictionary, else it will save it as a file
    if path is None:
        return dist_matrices
    else:
        np.savez(path + "dist_matrices.npz", **dist_matrices)


def get_pos_cell_labels(analysis_type, pheno=None, current_fov_neighborhood_data=None,
                        thresh=None, current_fov_channel_data=None, cell_labels=None, current_marker=None):
    """Based on the type of the analysis, the function finds positive labels that match the current phenotype or
    identifies cells with positive expression values for the current marker (greater than the marker threshold).

    Args:
        analysis_type (string): type of analysis, either "cluster" or "channel"
        pheno (string): the current cell phenotype
        current_fov_neighborhood_data (pandas df): data for the current patient
        thresh (int): current threshold for marker
        current_fov_channel_data (pandas df): expression data for column markers for current patient
        cell_labels (pandas df): the column of cell labels for current patient
        current_marker (string): the current marker that the positive labels are being found for
    Returns:
        mark1poslabels (list): all the positive labels"""

    if not np.isin(analysis_type, ("cluster", "channel")).all():
        raise ValueError("Incorrect analysis type")

    if analysis_type == "cluster":
        if pheno is None or current_fov_neighborhood_data is None:
            raise ValueError("Incorrect arguments passed for analysis type")
        # Subset only cells that are of the same phenotype
        pheno1posinds = current_fov_neighborhood_data["FlowSOM_ID"] == pheno
        # Get the cell labels of the cells of the phenotype
        mark1poslabels = current_fov_neighborhood_data.iloc[:, 1][pheno1posinds]
    else:
        if thresh is None or current_fov_channel_data is None or cell_labels is None or current_marker is None:
            raise ValueError("Incorrect arguments passed for analysis type")
        # Subset only cells that are positive for the given marker
        marker1posinds = current_fov_channel_data[current_marker] > thresh
        # Get the cell labels of the positive cells
        mark1poslabels = cell_labels[marker1posinds]
    return mark1poslabels


def compute_close_cell_num(dist_mat, dist_lim, num, analysis_type,
                           current_fov_data=None, current_fov_channel_data=None, cluster_ids=None,
                           thresh_vec=None):
    """Finds positive cell labels and creates matrix with counts for cells positive for corresponding markers.
    Computes close_num matrix for both Cell Label and Threshold spatial analyses.

    This function loops through all the included markers in the patient data and identifies cell labels positive for
    corresponding markers. It then subsets the distance matrix to only include these positive cells and records
    interactions based on whether cells are close to each other (within the dist_lim). It then stores the number of
    interactions in the index of close_num corresponding to both markers (for instance markers 1 and 2 would be in
    index [0, 1]).

    Args:
        dist_mat (np array): cells x cells matrix with the euclidian
            distance between centers of corresponding cells
        dist_lim (int): threshold for spatial enrichment distance proximity
        num (int): number of markers or cell phenotypes, based on analysis
        analysis_type (string): type of analysis, either cluster or channel
        current_fov_data (pandas df): data for specific patient in expression matrix
        current_fov_channel_data (pandas df): data of only column markers for Channel Analysis
        cluster_ids (pandas df): all the cell phenotypes in Cluster Analysis
        thresh_vec (numpy df): matrix of thresholds column for markers
        seed: the seed to set for randomized operations, useful for testing

    Returns:
        close_num (np array): marker x marker matrix with counts for cells
            positive for corresponding markers
        marker1_num: list of number of cell labels for marker 1
    """

    # Initialize variables

    cell_labels = []

    # Assign column names for subsetting (cell labels)
    cell_label_col = "cellLabelInImage"

    # Subset data based on analysis type
    if analysis_type == "channel":
        # Subsetting the column with the cell labels
        cell_labels = current_fov_data[cell_label_col]

    # Create close_num, marker1_num, and marker2_num
    close_num = np.zeros((num, num), dtype='int')

    mark1_num = []
    mark1poslabels = []

    # dist_mat_bin = np.zeros(dist_mat.shape, dtype='int')
    dist_mat_bin = np.zeros(dist_mat.shape, dtype='int')
    dist_mat_bin[dist_mat.values < dist_lim] = 1

    dist_mat_bin = xr.DataArray(dist_mat_bin,
                                coords=[dist_mat.dim_0.values, dist_mat.dim_1.values])

    for j in range(0, num):
        # Identify cell labels that are positive for respective markers or phenotypes, based on type of analysis
        if analysis_type == "cluster":
            mark1poslabels.append(get_pos_cell_labels(analysis_type, cluster_ids.iloc[j],
                                                      current_fov_data))
        else:
            mark1poslabels.append(get_pos_cell_labels(analysis_type, thresh=thresh_vec.iloc[j],
                                                      current_fov_channel_data=current_fov_channel_data,
                                                      cell_labels=cell_labels,
                                                      current_marker=current_fov_channel_data.columns[j]))
        mark1_num.append(len(mark1poslabels[j]))

    # iterating k from [j, end] cuts out 1/2 the steps (while symmetric)
    for j, m1n in enumerate(mark1_num):
        for k, m2n in enumerate(mark1_num[j:], j):
            close_num[j, k] = np.sum(dist_mat_bin.loc[
                np.array(mark1poslabels[j]),
                np.array(mark1poslabels[k])
            ].values)

            # symmetry :)
            close_num[k, j] = close_num[j, k]

    return close_num, mark1_num


def compute_close_cell_num_random(marker_nums, dist_mat, dist_lim, bootstrap_num):
    """Uses bootstrapping to permute cell labels randomly and records the number of close cells
    (within the dit_lim) in that random setup.

    Args
        marker_nums (np.array): list of cell counts of each marker type
        dist_mat (np.array): cells x cells matrix with the euclidian
            distance between centers of corresponding cells
        dist_lim (int): threshold for spatial enrichment distance proximity
        bootstrap_num (int): number of permutations

    Returns
        close_num_rand (np.array): random positive marker counts
            for every permutation in the bootstrap"""

    # Create close_num_rand
    close_num_rand = np.zeros((
        len(marker_nums), len(marker_nums), bootstrap_num), dtype='int')

        # dist_mat_bin = np.zeros(dist_mat.shape, dtype='int')
    dist_bin = np.zeros(dist_mat.shape, dtype='int')
    dist_bin[dist_mat.values < dist_lim] = 1

    dist_bin = xr.DataArray(dist_bin,
                            coords=[dist_mat.dim_0.values, dist_mat.dim_1.values])

    for j, m1n in enumerate(marker_nums):
        for k, m2n in enumerate(marker_nums[j:], j):
            close_num_rand[j, k, :] = np.sum(
                np.random.choice(dist_bin.values.flatten(), (m1n * m2n, bootstrap_num), True),
                axis=0
            )

            # symmetry :)
            close_num_rand[k, j, :] = close_num_rand[j, k, :]

    return close_num_rand


def calculate_enrichment_stats(close_num, close_num_rand):
    """Calculates z score and p values from spatial enrichment analysis.

    Args:
        close_num (np array): marker x marker matrix with counts for cells
            positive for corresponding markers
        close_num_rand (np array): random positive marker counts
            for every permutation in the bootstrap

    Returns:
        z (np array): z scores for corresponding markers
        muhat (np array): predicted mean values of close_num_rand random distribution
        sigmahat (np array): predicted standard deviation values of close_num_rand
            random distribution
        p (np array): p values for corresponding markers, for both positive
            and negative enrichment
        h (np array): matrix indicating whether
            corresponding marker interactions are significant
        adj_p (np array): fdh_br adjusted p values"""
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
            p_neg[j, k] = (1 + (np.sum(tmp < close_num[j, k]))) / (bootstrap_num + 1)

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


def compute_neighbor_counts(current_fov_neighborhood_data, dist_matrix, distlim, self_neighbor=True,
                            cell_label_col="cellLabelInImage"):
    """Calculates the number of neighbor phenotypes for each cell. The cell counts itself as a neighbor.

    Args:
        current_fov_neighborhood_data (pandas df): data for the current fov, including the cell labels, cell phenotypes, and cell phenotype ID
        dist_matrix (np array): cells x cells matrix with the euclidian
            distance between centers of corresponding cells
        distlim (int): threshold for spatial enrichment distance proximity
        self_neighbor (boolean): If true, cell counts itself as a neighbor in the analysis.
        cell_label_col (string): Column name with the cell labels
    Returns:
        counts (pandas df): data array with phenotype counts per cell
        freqs (pandas df): data array with phenotype frequencies of
            counts per phenotype/total phenotypes for each cell"""

    # TODO remove non-cell2cell lines (indices on the distance matrix not corresponding to cell labels)
    #  after our own inputs for functions are created
    # refine distance matrix to only cover cell labels in fov_data

    cell_dist_mat = dist_matrix.loc[
        np.array(current_fov_neighborhood_data[cell_label_col]),
        np.array(current_fov_neighborhood_data[cell_label_col])
    ].values

    # cell_dist_mat = np.take(dist_matrix, current_fov_neighborhood_data[cell_label_col] - 1, 0)
    # cell_dist_mat = np.take(cell_dist_mat, current_fov_neighborhood_data[cell_label_col] - 1, 1)

    # binarize distance matrix
    cell_dist_mat_bin = np.zeros(cell_dist_mat.shape)
    cell_dist_mat_bin[cell_dist_mat < distlim] = 1

    # default is that cell counts itself as a matrix
    if not self_neighbor:
        cell_dist_mat_bin[dist_matrix == 0] = 0

    # get num_neighbors for freqs
    num_neighbors = np.sum(cell_dist_mat_bin, axis=0)

    # create the 'phenotype has cell?' matrix, excluding non cell-label rows
    pheno_has_cell = pd.get_dummies(current_fov_neighborhood_data.iloc[:, 2]).to_numpy().T

    # dot binarized 'is neighbor?' matrix with pheno_has_cell to get counts
    counts = pheno_has_cell.dot(cell_dist_mat_bin).T

    # compute freqs with num_neighbors
    freqs = counts.T / num_neighbors

    return counts, freqs.T
