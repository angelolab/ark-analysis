from copy import deepcopy
from random import choices
from string import ascii_lowercase

import numpy as np
import pandas as pd
import xarray as xr

import ark.settings as settings
import synthetic_spatial_datagen

TEST_MARKERS = list('ABCDEFG')


def make_cell_table(num_cells, extra_cols=None):
    """ Generate a cell table with default column names for testing purposes.

    Args:
        num_cells (int):
            Number of rows (cells) in the cell table
        extra_cols (dict):
            Extra columns to add in the format ``{'Column_Name' : data_1D, ...}``

    Returns:
        pandas.DataFrame:
            A structural example of a cell table containing simulated marker expressions,
            cluster labels, centroid coordinates, and more.

    """
    # columns from regionprops extraction
    region_cols = [x for x in settings.REGIONPROPS_BASE if
                   x not in ['label', 'area', 'centroid']] + settings.REGIONPROPS_SINGLE_COMP
    region_cols += settings.REGIONPROPS_MULTI_COMP
    # consistent ordering of column names
    column_names = [settings.FOV_ID,
                    settings.PATIENT_ID,
                    settings.CELL_LABEL,
                    settings.CELL_TYPE,
                    settings.CELL_SIZE] + TEST_MARKERS + region_cols + ['centroid-0', 'centroid-1']

    if extra_cols is not None:
        column_names += list(extra_cols.values())

    # random filler data
    cell_data = pd.DataFrame(np.random.random(size=(num_cells, len(column_names))),
                             columns=column_names)
    # not-so-random filler data
    centroids = pd.DataFrame(np.array([(x, y) for x in range(1024) for y in range(1024)]))
    centroid_loc = np.random.choice(range(1024 ** 2), size=num_cells, replace=False)
    fields = [(settings.FOV_ID, choices(range(1, 5), k=num_cells)),
              (settings.PATIENT_ID, choices(range(1, 10), k=num_cells)),
              (settings.CELL_LABEL, list(range(num_cells))),
              (settings.CELL_TYPE, choices(ascii_lowercase, k=num_cells)),
              (settings.CELL_SIZE, np.random.uniform(100, 300, size=num_cells)),
              (settings.CENTROID_0, np.array(centroids.iloc[centroid_loc, 0])),
              (settings.CENTROID_1, np.array(centroids.iloc[centroid_loc, 1]))
              ]

    for name, col in fields:
        cell_data[name] = col

    return cell_data


# TODO: Use these below


EXCLUDE_CHANNELS = [
    "Background",
    "HH3",
    "summed_channel",
]

DEFAULT_COLUMNS_LIST = \
    [settings.CELL_SIZE] \
    + list(range(1, 24)) \
    + [
        settings.CELL_LABEL,
        'area',
        'eccentricity',
        'maj_axis_length',
        'min_axis_length',
        'perimeter',
        settings.FOV_ID,
        settings.CELL_TYPE,
    ]
list(map(
    DEFAULT_COLUMNS_LIST.__setitem__, [1, 14, 23], EXCLUDE_CHANNELS
))

DEFAULT_COLUMNS = dict(zip(range(33), DEFAULT_COLUMNS_LIST))


def create_test_extraction_data():
    """Generate hardcoded extraction test data

    Returns:
        tuple (numpy.ndarray, numpy.ndarray):

        - a sample segmentation mask
        - sample corresponding channel data
    """
    # first create segmentation masks
    cell_mask = np.zeros((1, 40, 40, 1), dtype='int16')
    cell_mask[:, 4:10, 4:10:, :] = 1
    cell_mask[:, 15:25, 20:30, :] = 2
    cell_mask[:, 27:32, 3:28, :] = 3
    cell_mask[:, 35:40, 15:22, :] = 5

    # then create channels data
    channel_data = np.zeros((1, 40, 40, 5), dtype="int16")
    channel_data[:, :, :, 0] = 1
    channel_data[:, :, :, 1] = 5
    channel_data[:, :, :, 2] = 5
    channel_data[:, :, :, 3] = 10
    channel_data[:, :, :, 4] = 0

    # cell1 is the only cell negative for channel 3
    channel_data[:, 4:10, 4:10, 3] = 0

    # cell2 is the only cell positive for channel 4
    channel_data[:, 15:25, 20:30, 4] = 10

    return cell_mask, channel_data


def _make_neighborhood_matrix():
    """Generate a sample neighborhood matrix

    Returns:
        pandas.DataFrame:
            a sample neighborhood matrix with three different populations,
            intended to test clustering
    """
    col_names = {0: settings.FOV_ID, 1: settings.CELL_LABEL, 2: settings.CELL_TYPE,
                 3: 'feature1', 4: 'feature2'}
    neighbor_counts = pd.DataFrame(np.zeros((200, 5)))
    neighbor_counts = neighbor_counts.rename(col_names, axis=1)

    neighbor_counts.iloc[0:100, 0] = "fov1"
    neighbor_counts.iloc[0:100, 1] = np.arange(100) + 1
    neighbor_counts.iloc[0:100, 2] = "cell_type"
    neighbor_counts.iloc[0:50, 3:5] = np.random.randint(low=0, high=10, size=(50, 2))
    neighbor_counts.iloc[50:100, 3:5] = np.random.randint(low=990, high=1000, size=(50, 2))

    neighbor_counts.iloc[100:200, 0] = "fov2"
    neighbor_counts.iloc[100:200, 1] = np.arange(100) + 1
    neighbor_counts.iloc[100:200, 2] = "cell_type"
    neighbor_counts.iloc[100:150, 3:5] = np.random.randint(low=990, high=1000, size=(50, 2))
    neighbor_counts.iloc[150:200, 3] = np.random.randint(low=0, high=10, size=50)
    neighbor_counts.iloc[150:200, 4] = np.random.randint(low=990, high=1000, size=50)

    return neighbor_counts


# TODO: it's very clunky and confusing to have to separate spatial analysis
# from spatial analysis utils synthetic data generation, here's an example
# of a function that I'd like to see be shared across both testing modules
# in the future
def _make_threshold_mat(in_utils):
    """Generate sample marker thresholds for testing channel enrichment

    Args:
        in_utils (bool):
            whether to generate for spatial_analysis or spatial_analysis_utils testing

    Returns:
        pandas.DataFrame:
            a sample marker threshold matrix for thresholding specifically for channel enrichment
    """

    thresh = pd.DataFrame(np.zeros((20, 2)), columns=["marker", "threshold"])
    thresh.iloc[:, 1] = .5

    if not in_utils:
        thresh.iloc[:, 0] = np.concatenate([np.arange(2, 14), np.arange(15, 23)])

        # spatial analysis should still be correct regardless of the marker threshold ordering
        thresh = thresh.sample(frac=1).reset_index(drop=True)

    return thresh


def _make_dist_mat_sa(enrichment_type, dist_lim):
    """Generate a sample distance matrix to test spatial_analysis

    Args:
        enrichment_type (str):
            whether to generate for positive, negative, or no enrichment
        dist_lim (int):
            the threshold to use for selecting entries in the distance matrix for enrichment

    Returns:
        xarray.DataArray:
            a sample distance matrix to use for testing spatial_analysis
    """

    if enrichment_type not in ["none", "positive", "negative"]:
        raise ValueError("enrichment_type must be none, positive, or negative")

    if enrichment_type == "none":
        # Create a 60 x 60 euclidian distance matrix of random values for no enrichment
        np.random.seed(0)
        rand_mat = np.random.randint(0, 200, size=(60, 60))
        np.fill_diagonal(rand_mat[:, :], 0)

        rand_mat = xr.DataArray(rand_mat,
                                coords=[np.arange(rand_mat.shape[0]) + 1,
                                        np.arange(rand_mat.shape[1]) + 1])

        fovs = ["fov8", "fov9"]
        mats = [rand_mat, rand_mat]
        rand_matrix = dict(zip(fovs, mats))

        return rand_matrix
    elif enrichment_type == "positive":
        # Create positive enrichment distance matrix where 10 cells mostly positive for marker 1
        # are located close in proximity to 10 cells mostly positive for marker 2.
        # Other included cells are not significantly positive for either marker and are located
        # far from the two positive populations.

        dist_mat_pos = synthetic_spatial_datagen.generate_test_dist_matrix(
            num_A=10, num_B=10, num_C=60, distr_AB=(int(dist_lim / 5), 1),
            distr_random=(int(dist_lim * 5), 1)
        )

        fovs = ["fov8", "fov9"]
        mats = [dist_mat_pos, dist_mat_pos]
        dist_mat_pos = dict(zip(fovs, mats))

        return dist_mat_pos
    elif enrichment_type == "negative":
        # This creates a distance matrix where there are two groups of cells significant for 2
        # different markers that are not located near each other (not within the dist_lim).

        dist_mat_neg = synthetic_spatial_datagen.generate_test_dist_matrix(
            num_A=20, num_B=20, num_C=20, distr_AB=(int(dist_lim * 5), 1),
            distr_random=(int(dist_lim / 5), 1)
        )

        fovs = ["fov8", "fov9"]
        mats = [dist_mat_neg, dist_mat_neg]
        dist_mat_neg = dict(zip(fovs, mats))

        return dist_mat_neg


def spoof_cell_table_from_labels(labels, cell_count=4, positive_population_ratio=1/4):
    """Generates example cell table from label images to test spatial_analysis batching

    Args:
        labels (xr.DataArray):
            data array with segmentation labels
        cell_count (int):
            number of cells per fov
        positive_population_ratio (float):
            fraction of cells per fov to assign unique trait to.  This is performed twice for two
            unique populations, so it must be smaller than 1/2.

    Returns:
        pandas.DataFrame:
            cell table which matches the provided label images
    """
    num_fovs = len(labels.fovs.values)

    if positive_population_ratio > 1/2:
        raise ValueError('population_ratio must be less than 1/2')

    cell_table = pd.DataFrame(np.zeros((num_fovs * cell_count, 33)))
    for i, fov in enumerate(labels.fovs.values):
        fov_rows = np.arange(start=i * cell_count, stop=(i + 1) * cell_count)

        pop_count = int(cell_count * positive_population_ratio)

        cell_table.loc[fov_rows, 30] = fov

        cell_table.loc[fov_rows, 24] = np.unique(labels.loc[fov, :, :, :])[1:]

        # create unique populations
        cell_table.iloc[fov_rows[0:pop_count], (i % 2) + 2] = 1
        cell_table.iloc[fov_rows[pop_count:2 * pop_count], ((i + 1) % 2) + 2] = 1

        cell_table.iloc[fov_rows[0:pop_count], 31] = 1 + (i % 2)
        cell_table.iloc[fov_rows[0:pop_count], 32] = f"Pheno{1 + (i % 2)}"
        cell_table.iloc[fov_rows[pop_count:2 * pop_count], 31] = 1 + ((i + 1) % 2)
        cell_table.iloc[fov_rows[pop_count:2 * pop_count], 32] = f"Pheno{1 + ((i + 1) % 2)}"

    cell_table = cell_table.rename(DEFAULT_COLUMNS, axis=1)

    cell_table.loc[cell_table.iloc[:, 31] == 0, settings.CELL_TYPE] = "Pheno3"

    return cell_table


def _make_expression_mat_sa(enrichment_type):
    """Generate a sample expression matrix to test spatial_analysis

    Args:
        enrichment_type (str):
            whether to generate for positive, negative, or no enrichment

    Returns:
        pandas.DataFrame:
            an expression matrix with cell labels and patient labels
    """

    if enrichment_type not in ["none", "positive", "negative"]:
        raise ValueError("enrichment_type must be none, positive, or negative")

    if enrichment_type == "none":
        all_data = pd.DataFrame(np.zeros((120, 32)))
        # Assigning values to the patient label and cell label columns
        # We create data for two fovs, with the second fov being the same as the first but the
        # cell expression data for marker 1 and marker 2 are inverted. cells 0-59 are fov8 and
        # cells 60-119 are fov9
        all_data.loc[0:59, 30] = "fov8"
        all_data.loc[60:, 30] = "fov9"
        all_data.loc[0:59, 24] = np.arange(60) + 1
        all_data.loc[60:, 24] = np.arange(60) + 1
        # We create two populations of 20 cells, each positive for different marker (index 2 and 3)
        all_data.iloc[0:20, 2] = 1
        all_data.iloc[20:40, 3] = 1

        all_data.iloc[60:80, 3] = 1
        all_data.iloc[80:100, 2] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data.iloc[0:20, 31] = "Pheno1"
        all_data.iloc[60:80, 31] = "Pheno2"

        all_data.iloc[20:40, 31] = "Pheno2"
        all_data.iloc[80:100, 31] = "Pheno1"

        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data = all_data.rename(DEFAULT_COLUMNS, axis=1)

        all_patient_data.loc[all_patient_data.iloc[:, 31] == 0, settings.CELL_TYPE] = "Pheno3"
        return all_patient_data
    elif enrichment_type == "positive":
        all_data_pos = pd.DataFrame(np.zeros((160, 32)))
        # Assigning values to the patient label and cell label columns
        all_data_pos.loc[0:79, 30] = "fov8"
        all_data_pos.loc[80:, 30] = "fov9"
        all_data_pos.loc[0:79, 24] = np.arange(80) + 1
        all_data_pos.loc[80:, 24] = np.arange(80) + 1
        # We create 8 cells positive for column index 2, and 8 cells positive for column index 3.
        # These are within the dist_lim in dist_mat_pos (positive enrichment distance matrix).
        all_data_pos.iloc[0:8, 2] = 1
        all_data_pos.iloc[10:18, 3] = 1

        all_data_pos.iloc[80:88, 3] = 1
        all_data_pos.iloc[90:98, 2] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data_pos.iloc[0:8, 31] = "Pheno1"
        all_data_pos.iloc[80:88, 31] = "Pheno2"

        all_data_pos.iloc[10:18, 31] = "Pheno2"
        all_data_pos.iloc[90:98, 31] = "Pheno1"
        # We create 4 cells in column index 2 and column index 3 that are also positive
        # for their respective markers.
        all_data_pos.iloc[28:32, 2] = 1
        all_data_pos.iloc[32:36, 3] = 1
        all_data_pos.iloc[108:112, 3] = 1
        all_data_pos.iloc[112:116, 2] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data_pos.iloc[28:32, 31] = "Pheno1"
        all_data_pos.iloc[108:112, 31] = "Pheno2"

        all_data_pos.iloc[32:36, 31] = "Pheno2"
        all_data_pos.iloc[112:116, 31] = "Pheno1"

        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_pos = all_data_pos.rename(DEFAULT_COLUMNS, axis=1)

        all_patient_data_pos.loc[all_patient_data_pos.iloc[:, 31] == 0,
                                 settings.CELL_TYPE] = "Pheno3"
        return all_patient_data_pos
    elif enrichment_type == "negative":
        all_data_neg = pd.DataFrame(np.zeros((120, 32)))
        # Assigning values to the patient label and cell label columns
        all_data_neg.loc[0:59, 30] = "fov8"
        all_data_neg.loc[60:, 30] = "fov9"
        all_data_neg.loc[0:59, 24] = np.arange(60) + 1
        all_data_neg.loc[60:, 24] = np.arange(60) + 1
        # We create two groups of 20 cells positive for marker 1 (in column index 2)
        # and marker 2 (in column index 3) respectively.
        # The two populations are not within the dist_lim in dist_mat_neg
        all_data_neg.iloc[0:20, 2] = 1
        all_data_neg.iloc[20:40, 3] = 1

        all_data_neg.iloc[60:80, 3] = 1
        all_data_neg.iloc[80:100, 2] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data_neg.iloc[0:20, 31] = "Pheno1"
        all_data_neg.iloc[60:80, 31] = "Pheno2"

        all_data_neg.iloc[20:40, 31] = "Pheno2"
        all_data_neg.iloc[80:100, 31] = "Pheno1"

        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_neg = all_data_neg.rename(DEFAULT_COLUMNS, axis=1)

        all_patient_data_neg.loc[all_patient_data_neg.iloc[:, 31] == 0,
                                 settings.CELL_TYPE] = "Pheno3"
        return all_patient_data_neg


def _make_dist_exp_mats_spatial_test(enrichment_type, dist_lim):
    """Generate example expression and distance matrices for testing spatial_analysis

    Args:
        enrichment_type (str):
            whether to generate for positive, negative, or no enrichment
        dist_lim (int):
            the threshold to use for selecting entries in the distance matrix for enrichment

    Returns:
        tuple (pandas.DataFrame, xarray.DataArray):

        - a sample expression matrix
        - a sample distance matrix
    """

    all_data = _make_expression_mat_sa(enrichment_type=enrichment_type)
    dist_mat = _make_dist_mat_sa(enrichment_type=enrichment_type, dist_lim=dist_lim)

    return all_data, dist_mat


def _make_context_dist_exp_mats_spatial_test(dist_lim):
    all_data = _make_expression_mat_sa("none")
    dist_mat = _make_dist_mat_sa("none", dist_lim)

    all_data['context_col'] = (['context_A', ] * 60) + (['context_B', ] * 60)
    return all_data, dist_mat


def _make_dist_exp_mats_dist_feature_spatial_test(dist_lim):
    all_data = _make_expression_mat_sa("none")
    dist_mat = _make_dist_mat_sa("none", dist_lim)

    all_data['dist_whole_cell'] = (dist_lim / 2) * np.ones(all_data.shape[0])
    return all_data, dist_mat


def _make_dist_mat_sa_utils():
    """Generate a sample distance matrix to test spatial_analysis_utils

    Returns:
        xarray.DataArray:
            a sample distance matrix to use for testing spatial_analysis_utils
    """

    dist_mat = np.zeros((10, 10))
    np.fill_diagonal(dist_mat, 0)

    # Create distance matrix where cells positive for marker 1 and 2 are within the dist_lim of
    # each other, but not the other groups. This is repeated for cells positive for marker 3 and 4,
    # and for cells positive for marker 5.
    dist_mat[1:4, 0] = 50
    dist_mat[0, 1:4] = 50
    dist_mat[4:9, 0] = 200
    dist_mat[0, 4:9] = 200
    dist_mat[9, 0] = 500
    dist_mat[0, 9] = 500
    dist_mat[2:4, 1] = 50
    dist_mat[1, 2:4] = 50
    dist_mat[4:9, 1] = 150
    dist_mat[1, 4:9] = 150
    dist_mat[9, 1:9] = 200
    dist_mat[1:9, 9] = 200
    dist_mat[3, 2] = 50
    dist_mat[2, 3] = 50
    dist_mat[4:9, 2] = 150
    dist_mat[2, 4:9] = 150
    dist_mat[4:9, 3] = 150
    dist_mat[3, 4:9] = 150
    dist_mat[5:9, 4] = 50
    dist_mat[4, 5:9] = 50
    dist_mat[6:9, 5] = 50
    dist_mat[5, 6:9] = 50
    dist_mat[7:9, 6] = 50
    dist_mat[6, 7:9] = 50
    dist_mat[8, 7] = 50
    dist_mat[7, 8] = 50

    # add some randomization to the ordering
    coords_in_order = np.arange(dist_mat.shape[0])
    coords_permuted = deepcopy(coords_in_order)
    np.random.shuffle(coords_permuted)
    dist_mat = dist_mat[np.ix_(coords_permuted, coords_permuted)]

    # we have to 1-index coords because people will be labeling their cells 1-indexed
    coords_dist_mat = [coords_permuted + 1, coords_permuted + 1]
    dist_mat = xr.DataArray(dist_mat, coords=coords_dist_mat)

    return dist_mat


def _make_expression_mat_sa_utils():
    """Generate a sample expression matrix to test spatial_analysis_utils

    Returns:
        pandas.DataFrame:
            an expression matrix with cell labels and patient labels
    """

    # Create example all_patient_data cell expression matrix
    all_data = pd.DataFrame(np.zeros((10, 32)))

    # Assigning values to the patient label and cell label columns
    all_data[30] = "fov8"
    all_data[24] = np.arange(len(all_data[1])) + 1

    colnames = {
        0: settings.CELL_SIZE,
        24: settings.CELL_LABEL,
        30: settings.FOV_ID,
        31: settings.CELL_TYPE,
    }
    all_data = all_data.rename(colnames, axis=1)

    # Create 4 cells positive for marker 1 and 2, 5 cells positive for markers 3 and 4,
    # and 1 cell positive for marker 5
    all_data.iloc[0:4, 2] = 1
    all_data.iloc[0:4, 3] = 1
    all_data.iloc[4:9, 5] = 1
    all_data.iloc[4:9, 6] = 1
    all_data.iloc[9, 7] = 1
    all_data.iloc[9, 8] = 1

    # 4 cells assigned one phenotype, 5 cells assigned another phenotype,
    # and the last cell assigned a different phenotype
    all_data.iloc[0:4, 31] = "Pheno1"
    all_data.iloc[4:9, 31] = "Pheno2"
    all_data.iloc[9, 31] = "Pheno3"

    return all_data


def _make_dist_exp_mats_spatial_utils_test():
    """Generate example expression and distance matrices for testing spatial_analysis_utils

    Returns:
        tuple (pandas.DataFrame, xarray.DataArray):

        - a sample expression matrix
        - a sample distance matrix
    """

    all_data = _make_expression_mat_sa_utils()
    dist_mat = _make_dist_mat_sa_utils()

    return all_data, dist_mat


def generate_sample_fov_tiling_entry(coord, name):
    """Generates a sample fov entry to put in a sample fovs list for tiling

    Args:
        coord (tuple):
            Defines the starting x and y point for the fov
        name (str):
            Defines the name of the fov

    Returns:
        dict:
            An entry to be placed in the fovs list with provided coordinate and name
    """

    sample_fov_tiling_entry = {
        "scanCount": 1,
        "centerPointMicrons": {
            "x": coord[0],
            "y": coord[1]
        },
        "timingChoice": 7,
        "frameSizePixels": {
            "width": 2048,
            "height": 2048
        },
        "imagingPreset": {
            "preset": "Normal",
            "aperture": "2",
            "displayName": "Fine",
            "defaults": {
                "timingChoice": 7
            }
        },
        "sectionId": 8201,
        "slideId": 5931,
        "name": name,
        "timingDescription": "1 ms"
    }

    return sample_fov_tiling_entry


def generate_sample_fovs_list(fov_coords, fov_names):
    """Generate a sample dictionary of fovs for tiling

    Args:
        fov_coords (list):
            A list of tuples listing the starting x and y coordinates of each fov
        fov_names (list):
            A list of strings identifying the name of each fov

    Returns:
        dict:
            A dummy fovs list with starting x and y set to the provided coordinates and name
    """

    sample_fovs_list = {
        "exportDateTime": "2021-03-12T19:02:37.920Z",
        "fovFormatVersion": "1.5",
        "fovs": []
    }

    for coord, name in zip(fov_coords, fov_names):
        sample_fovs_list["fovs"].append(
            generate_sample_fov_tiling_entry(coord, name)
        )

    return sample_fovs_list
