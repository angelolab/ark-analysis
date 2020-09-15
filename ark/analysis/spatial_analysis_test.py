import pytest
import numpy as np
import pandas as pd
import xarray as xr

from ark.analysis import spatial_analysis
from ark.utils import synthetic_spatial_datagen


def _make_threshold_mat():
    thresh = pd.DataFrame(np.zeros((20, 2)))
    thresh.iloc[:, 1] = .5
    thresh.iloc[:, 0] = np.arange(20) + 2
    return thresh


def _make_distance_matrix(enrichment_type, dist_lim):
    # Make a distance matrix for no enrichment, positive enrichment, and negative enrichment

    if enrichment_type == "none":
        # Create a 60 x 60 euclidian distance matrix of random values for no enrichment
        np.random.seed(0)
        rand_mat = np.random.randint(0, 200, size=(60, 60))
        np.fill_diagonal(rand_mat[:, :], 0)

        rand_mat = xr.DataArray(rand_mat,
                                coords=[np.arange(rand_mat.shape[0]) + 1,
                                        np.arange(rand_mat.shape[1]) + 1])

        fovs = ["Point8", "Point9"]
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

        fovs = ["Point8", "Point9"]
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

        fovs = ["Point8", "Point9"]
        mats = [dist_mat_neg, dist_mat_neg]
        dist_mat_neg = dict(zip(fovs, mats))

        return dist_mat_neg


def _make_expression_matrix(enrichment_type):
    # Create the expression matrix with cell labels and patient labels for no enrichment,
    # positive enrichment, and negative enrichment.

    # Column names for columns that are not markers (columns to be excluded)
    excluded_colnames = {0: 'cell_size', 1: 'Background', 14: "HH3",
                         23: "summed_channel", 24: "cellLabelInImage", 25: "area",
                         26: "eccentricity", 27: "major_axis_length", 28: "minor_axis_length",
                         29: "perimeter", 30: "SampleID", 31: "FlowSOM_ID", 32: "cell_type"}

    if enrichment_type == "none":
        all_data = pd.DataFrame(np.zeros((120, 33)))
        # Assigning values to the patient label and cell label columns
        # We create data for two fovs, with the second fov being the same as the first but the
        # cell expression data for marker 1 and marker 2 are inverted. cells 0-59 are Point8 and
        # cells 60-119 are Point9
        all_data.loc[0:59, 30] = "Point8"
        all_data.loc[60:, 30] = "Point9"
        all_data.loc[0:59, 24] = np.arange(60) + 1
        all_data.loc[60:, 24] = np.arange(60) + 1
        # We create two populations of 20 cells, each positive for different marker (index 2 and 3)
        all_data.iloc[0:20, 2] = 1
        all_data.iloc[20:40, 3] = 1

        all_data.iloc[60:80, 3] = 1
        all_data.iloc[80:100, 2] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data.iloc[0:20, 31] = 1
        all_data.iloc[0:20, 32] = "Pheno1"
        all_data.iloc[60:80, 31] = 2
        all_data.iloc[60:80, 32] = "Pheno2"

        all_data.iloc[20:40, 31] = 2
        all_data.iloc[20:40, 32] = "Pheno2"
        all_data.iloc[80:100, 31] = 1
        all_data.iloc[80:100, 32] = "Pheno1"

        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data = all_data.rename(excluded_colnames, axis=1)

        all_patient_data.loc[all_patient_data.iloc[:, 31] == 0, "cell_type"] = "Pheno3"
        return all_patient_data
    elif enrichment_type == "positive":
        all_data_pos = pd.DataFrame(np.zeros((160, 33)))
        # Assigning values to the patient label and cell label columns
        all_data_pos.loc[0:79, 30] = "Point8"
        all_data_pos.loc[80:, 30] = "Point9"
        all_data_pos.loc[0:79, 24] = np.arange(80) + 1
        all_data_pos.loc[80:, 24] = np.arange(80) + 1
        # We create 8 cells positive for column index 2, and 8 cells positive for column index 3.
        # These are within the dist_lim in dist_mat_pos (positive enrichment distance matrix).
        all_data_pos.iloc[0:8, 2] = 1
        all_data_pos.iloc[10:18, 3] = 1

        all_data_pos.iloc[80:88, 3] = 1
        all_data_pos.iloc[90:98, 2] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data_pos.iloc[0:8, 31] = 1
        all_data_pos.iloc[0:8, 32] = "Pheno1"
        all_data_pos.iloc[80:88, 31] = 2
        all_data_pos.iloc[80:88, 32] = "Pheno2"

        all_data_pos.iloc[10:18, 31] = 2
        all_data_pos.iloc[10:18, 32] = "Pheno2"
        all_data_pos.iloc[90:98, 31] = 1
        all_data_pos.iloc[90:98, 32] = "Pheno1"
        # We create 4 cells in column index 2 and column index 3 that are also positive
        # for their respective markers.
        all_data_pos.iloc[28:32, 2] = 1
        all_data_pos.iloc[32:36, 3] = 1
        all_data_pos.iloc[108:112, 3] = 1
        all_data_pos.iloc[112:116, 2] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data_pos.iloc[28:32, 31] = 1
        all_data_pos.iloc[28:32, 32] = "Pheno1"
        all_data_pos.iloc[108:112, 31] = 2
        all_data_pos.iloc[108:112, 32] = "Pheno2"

        all_data_pos.iloc[32:36, 31] = 2
        all_data_pos.iloc[32:36, 32] = "Pheno2"
        all_data_pos.iloc[112:116, 31] = 1
        all_data_pos.iloc[112:116, 32] = "Pheno1"

        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_pos = all_data_pos.rename(excluded_colnames, axis=1)

        all_patient_data_pos.loc[all_patient_data_pos.iloc[:, 31] == 0, "cell_type"] = "Pheno3"
        return all_patient_data_pos
    elif enrichment_type == "negative":
        all_data_neg = pd.DataFrame(np.zeros((120, 33)))
        # Assigning values to the patient label and cell label columns
        all_data_neg.loc[0:59, 30] = "Point8"
        all_data_neg.loc[60:, 30] = "Point9"
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
        all_data_neg.iloc[0:20, 31] = 1
        all_data_neg.iloc[0:20, 32] = "Pheno1"
        all_data_neg.iloc[60:80, 31] = 2
        all_data_neg.iloc[60:80, 32] = "Pheno2"

        all_data_neg.iloc[20:40, 31] = 2
        all_data_neg.iloc[20:40, 32] = "Pheno2"
        all_data_neg.iloc[80:100, 31] = 1
        all_data_neg.iloc[80:100, 32] = "Pheno1"

        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_neg = all_data_neg.rename(excluded_colnames, axis=1)

        all_patient_data_neg.loc[all_patient_data_neg.iloc[:, 31] == 0, "cell_type"] = "Pheno3"
        return all_patient_data_neg


def test_calculate_channel_spatial_enrichment():

    dist_lim = 100

    excluded_colnames = ["cell_size", "Background", "HH3",
                         "summed_channel", "cellLabelInImage", "area",
                         "eccentricity", "major_axis_length", "minor_axis_length",
                         "perimeter", "SampleID", "FlowSOM_ID", "cell_type"]

    # Test z and p values
    marker_thresholds = _make_threshold_mat()

    # Positive enrichment with direct matrix initialization
    all_data_pos = _make_expression_matrix(enrichment_type="positive")
    dist_mat_pos_direct = _make_distance_matrix(enrichment_type="positive", dist_lim=dist_lim)

    _, stats_pos = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_pos_direct, marker_thresholds, all_data_pos,
            excluded_colnames=excluded_colnames, bootstrap_num=100,
            dist_lim=dist_lim)

    # Test both Point8 and Point9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for positive
    # enrichment as tested against a random set of distances between centroids
    assert stats_pos.loc["Point8", "p_pos", 2, 3] < .05
    assert stats_pos.loc["Point8", "p_neg", 2, 3] > .05
    assert stats_pos.loc["Point8", "z", 2, 3] > 0

    assert stats_pos.loc["Point9", "p_pos", 3, 2] < .05
    assert stats_pos.loc["Point9", "p_neg", 3, 2] > .05
    assert stats_pos.loc["Point9", "z", 3, 2] > 0

    # Negative enrichment with direct matrix initialization
    all_data_neg = _make_expression_matrix("negative")
    dist_mat_neg_direct = _make_distance_matrix("negative", dist_lim=dist_lim)

    _, stats_neg = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_neg_direct, marker_thresholds, all_data_neg,
            excluded_colnames=excluded_colnames, bootstrap_num=100,
            dist_lim=dist_lim)

    # Test both Point8 and Point9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for negative
    # enrichment as tested against a random set of distances between centroids
    assert stats_neg.loc["Point8", "p_neg", 2, 3] < .05
    assert stats_neg.loc["Point8", "p_pos", 2, 3] > .05
    assert stats_neg.loc["Point8", "z", 2, 3] < 0

    assert stats_neg.loc["Point9", "p_neg", 3, 2] < .05
    assert stats_neg.loc["Point9", "p_pos", 3, 2] > .05
    assert stats_neg.loc["Point9", "z", 3, 2] < 0

    # No enrichment
    all_data_no_enrich = _make_expression_matrix("none")
    dist_mat_no_enrich = _make_distance_matrix("none", dist_lim=dist_lim)

    _, stats_no_enrich = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_no_enrich, marker_thresholds, all_data_no_enrich,
            excluded_colnames=excluded_colnames, bootstrap_num=100,
            dist_lim=dist_lim)
    # Test both Point8 and Point9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for no enrichment
    # as tested against a random set of distances between centroids
    assert stats_no_enrich.loc["Point8", "p_pos", 2, 3] > .05
    assert stats_no_enrich.loc["Point8", "p_neg", 2, 3] > .05
    assert abs(stats_no_enrich.loc["Point8", "z", 2, 3]) < 2

    assert stats_no_enrich.loc["Point9", "p_pos", 3, 2] > .05
    assert stats_no_enrich.loc["Point9", "p_neg", 3, 2] > .05
    assert abs(stats_no_enrich.loc["Point9", "z", 3, 2]) < 2

    # error checking
    with pytest.raises(ValueError):
        # attempt to exclude a column name that doesn't appear in the expression matrix
        _, stats_no_enrich = \
            spatial_analysis.calculate_channel_spatial_enrichment(
                dist_mat_no_enrich, marker_thresholds, all_data_no_enrich,
                excluded_colnames=["bad_excluded_col_name"], bootstrap_num=100,
                dist_lim=dist_lim)

    with pytest.raises(ValueError):
        # attempt to include fovs that do not exist
        _, stat_no_enrich = \
            spatial_analysis.calculate_channel_spatial_enrichment(
                dist_mat_no_enrich, marker_thresholds, all_data_no_enrich,
                excluded_colnames=excluded_colnames, included_fovs=[1, 100000],
                bootstrap_num=100, dist_lim=dist_lim)

    with pytest.raises(ValueError):
        # attempt to include marker thresholds that do not exist
        bad_marker_thresholds = pd.DataFrame(np.zeros((20, 2)))
        bad_marker_thresholds.iloc[:, 1] = .5
        bad_marker_thresholds.iloc[:, 0] = np.arange(10000, 10020) + 2

        _, stat_no_enrich = \
            spatial_analysis.calculate_channel_spatial_enrichment(
                dist_mat_no_enrich, bad_marker_thresholds, all_data_no_enrich,
                excluded_colnames=excluded_colnames, bootstrap_num=100,
                dist_lim=dist_lim)


def test_calculate_cluster_spatial_enrichment():
    # Test z and p values
    dist_lim = 100

    # Positive enrichment with direct matrix initialization
    all_data_pos = _make_expression_matrix(enrichment_type="positive")
    dist_mat_pos_direct = _make_distance_matrix(enrichment_type="positive", dist_lim=dist_lim)

    _, stats_pos = \
        spatial_analysis.calculate_cluster_spatial_enrichment(
            all_data_pos, dist_mat_pos_direct,
            bootstrap_num=100, dist_lim=dist_lim)
    # Test both Point8 and Point9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for positive
    # enrichment as tested against a random set of distances between centroids
    assert stats_pos.loc["Point8", "p_pos", "Pheno1", "Pheno2"] < .05
    assert stats_pos.loc["Point8", "p_neg", "Pheno1", "Pheno2"] > .05
    assert stats_pos.loc["Point8", "z", "Pheno1", "Pheno2"] > 0

    assert stats_pos.loc["Point9", "p_pos", "Pheno2", "Pheno1"] < .05
    assert stats_pos.loc["Point9", "p_neg", "Pheno2", "Pheno1"] > .05
    assert stats_pos.loc["Point9", "z", "Pheno2", "Pheno1"] > 0

    # Negative enrichment with direct matrix initialization
    all_data_neg = _make_expression_matrix("negative")
    dist_mat_neg_direct = _make_distance_matrix(enrichment_type="negative", dist_lim=dist_lim)

    _, stats_neg = \
        spatial_analysis.calculate_cluster_spatial_enrichment(
            all_data_neg, dist_mat_neg_direct,
            bootstrap_num=100, dist_lim=dist_lim)
    # Test both Point8 and Point9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for negative
    # enrichment as tested against a random set of distances between centroids
    assert stats_neg.loc["Point8", "p_neg", "Pheno1", "Pheno2"] < .05
    assert stats_neg.loc["Point8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert stats_neg.loc["Point8", "z", "Pheno1", "Pheno2"] < 0

    assert stats_neg.loc["Point9", "p_neg", "Pheno2", "Pheno1"] < .05
    assert stats_neg.loc["Point9", "p_pos", "Pheno2", "Pheno1"] > .05
    assert stats_neg.loc["Point9", "z", "Pheno2", "Pheno1"] < 0

    # No enrichment
    all_data_no_enrich = _make_expression_matrix("none")
    dist_mat_no_enrich = _make_distance_matrix("none", dist_lim=dist_lim)

    _, stats_no_enrich = \
        spatial_analysis.calculate_cluster_spatial_enrichment(
            all_data_no_enrich, dist_mat_no_enrich,
            bootstrap_num=100, dist_lim=dist_lim)
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for no enrichment
    # as tested against a random set of distances between centroids
    assert stats_no_enrich.loc["Point8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert stats_no_enrich.loc["Point8", "p_neg", "Pheno1", "Pheno2"] > .05
    assert abs(stats_no_enrich.loc["Point8", "z", "Pheno1", "Pheno2"]) < 2

    assert stats_no_enrich.loc["Point8", "p_pos", "Pheno2", "Pheno1"] > .05
    assert stats_no_enrich.loc["Point8", "p_neg", "Pheno2", "Pheno1"] > .05
    assert abs(stats_no_enrich.loc["Point8", "z", "Pheno2", "Pheno1"]) < 2

    # error checking
    with pytest.raises(ValueError):
        # attempt to include fovs that do not exist
        _, stats_no_enrich = \
            spatial_analysis.calculate_cluster_spatial_enrichment(
                all_data_no_enrich, dist_mat_no_enrich, included_fovs=[1, 100000],
                bootstrap_num=100, dist_lim=dist_lim)


def test_create_neighborhood_matrix():
    # get positive expression and distance matrices
    all_data_pos = _make_expression_matrix("positive")
    dist_mat_pos = _make_distance_matrix("positive", dist_lim=51)

    counts, freqs = spatial_analysis.create_neighborhood_matrix(
        all_data_pos, dist_mat_pos, distlim=51
    )

    # Test the counts values for both fovs
    assert (counts.loc[:9, "Pheno2"] == 8).all()
    assert (counts.loc[10:19, "Pheno3"] == 8).all()

    assert (counts.loc[80:89, "Pheno3"] == 8).all()
    assert (counts.loc[90:99, "Pheno1"] == 8).all()

    # error checking
    with pytest.raises(ValueError):
        # attempt to include fovs that do not exist
        counts, freqs = spatial_analysis.create_neighborhood_matrix(
            all_data_pos, dist_mat_pos, included_fovs=[1, 100000], distlim=51
        )
