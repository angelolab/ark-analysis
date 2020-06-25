import pandas as pd
import xarray as xr
import numpy as np
from segmentation.utils import spatial_analysis_utils
import importlib
importlib.reload(spatial_analysis_utils)

EX_COLNAMES = ["cell_size", "Background", "HH3",
               "summed_channel", "label", "area",
               "eccentricity", "major_axis_length", "minor_axis_length",
               "perimeter", "fov"]

COORDS_MARKERS = ["z", "muhat", "sigmahat", "p_pos", "p_neg", "h", "p_adj"]

DIMS_MARKERS = ["points", "stats", "marker1", "marker2"]


# Erin's Data Inputs-Threshold

# cell_array = pd.read_csv("/Users/jaiveersingh/Downloads/SpatialEn"
#                          "richment/granA_cellpheno_CS-asinh-norm_revised.csv")
# marker_thresholds = pd.read_csv("/Users/jaiveersingh/Downloads/Sp"
#                                 "atialEnrichment/markerThresholds.csv")
# dist_matrix = np.asarray(pd.read_csv("/Users/jaiveersingh/Documen"
#                                      "ts/MATLAB/distancesMat5.csv",
#                                      header=None))

# Erin's Data Inputs-Phenotype
# all_patient_data = pd.read_csv("/Users/jaiveersingh/Desktop/granA_cellpheno_CS-asinh-norm_matlab_revised.csv")
# pheno = pd.read_csv("/Users/jaiveersingh/Downloads/CellType_GlobalSpatialEnrichment/cellpheno_numkey.csv")
# dist_mat = np.asarray(pd.read_csv("/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header=None))

def generate_channel_spatial_enrichment_data(dist_matrices, data_markers, marker_num,
                                             marker_titles, all_data, thresh_vec,
                                             num_fovs, fovs=None, fov_col="SampleID",
                                             dist_lim=100, bootstrap_num=1000):

    """Generate the values array and stats matrix used by calculate_channel_spatial_enrichment

    Args:
        dist_matrices: A dictionary that contains a cells x cells matrix with the euclidian
            distance between centers of corresponding cells for every fov.
        data_markers: data including points, cell labels, and
            cell expression matrix for all markers including only desired columns.
        marker_num: length of the marker list.
        marker_titles: a list of all the markers.
        all_data: data including points, cell labels, and
            cell expression matrix for all markers.
        thresh_vec: a subset of the threshold matrix including only the
            column with the desired threshold values.
        num_fovs: the number of fovs to iterate over.
        fovs: patient labels to include in analysis. Default all labels used as computed
            in calculate_channel_spatial_enrichment.
        fov_col: the list of column names we wish to extract from fovs. Default SampleID.
        dist_lim: cell proximity threshold. Default 100.
        bootstrap_num: number of permutations for bootstrap. Default 1000.
    """

    # Initialize the values list
    values = []

    # Create stats Xarray with the dimensions (points, stats variables, number of markers, number of markers)
    stats_raw_data = np.zeros((num_fovs, 7, marker_num, marker_num))
    coords = [fovs, COORDS_MARKERS, marker_titles, marker_titles]
    dims = DIMS_MARKERS

    stats = xr.DataArray(stats_raw_data, coords=coords, dims=dims)

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

    if excluded_colnames is None:
        excluded_colnames = EX_COLNAMES

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

    # Subsetting threshold matrix to only include column with threshold values
    thresh_vec = marker_thresholds.iloc[:, 1]

    # generate the values list and the stats Xarray
    values, stats = generate_channel_spatial_enrichment_data(
        dist_matrices, data_markers, marker_num, marker_titles, all_data,
        thresh_vec, num_fovs, fovs, fov_col, dist_lim, bootstrap_num)

    return values, stats


def generate_cluster_spatial_enrichment_data(dist_mats, pheno_titles, pheno_num,
                                             pheno_codes, fov_cluster_data, num_fovs,
                                             fovs=None, bootstrap_num=1000, dist_lim=100):
    """Generate the values array and stats matrix used by calculate_cluster_spatial_enrichment

    Args:
        dist_mats: A dictionary that contains a cells x cells matrix with the euclidian
            distance between centers of corresponding cells for every fov.
        pheno_titles: the names of the cell phenotypes.
        pheno_codes: the codes corresponding to the columns of cell phenotype data.
        fov_cluster_data: a subset matrix of all_data from calculate_cluster_spatial_enrichment
            that includes only the desired columns.
        num_fovs: the number of fovs to iterate over.
        fovs: patient labels to include in analysis. Default all labels used as computed
            in calculate_cluster_spatial_enrichment.
        bootstrap_num: number of permutations for bootstrap. Default 1000.
        dist_lim: cell proximity threshold. Default 100.
    """

    # Initialize the values list
    values = []

    # Create stats Xarray with the dimensions (points, stats variables, number of markers, number of markers)
    stats_raw_data = np.zeros((num_fovs, 7, pheno_num, pheno_num))
    coords = [fovs, COORDS_MARKERS, pheno_titles, pheno_titles]
    dims = DIMS_MARKERS
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

    values, stats = generate_cluster_spatial_enrichment_data(
        dist_mats, pheno_titles, pheno_num, pheno_codes, fov_cluster_data, num_fovs,
        fovs, bootstrap_num, dist_lim)

    return values, stats


















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

            # if sigmahat[j, k] == 0:
            #     print("Tmp array is: ")
            #     print(tmp)
            #     print('Because sigmahat[j, k] is 0, z[j, k] is now: %.5f' % z[j, k])
            #     print('Close num[j, k] is %.5f' % close_num[j, k])
            #     print('P_pos is %.5f' % p_pos[j, k])
            #     print('P_neg is %.5f' % p_neg[j, k])

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
