import numpy as np
import pandas as pd
import random
from segmentation.utils import spatial_analysis
import importlib
importlib.reload(spatial_analysis)


def make_threshold_mat():
    thresh = pd.DataFrame(np.zeros((20, 2)))
    thresh.iloc[:, 1] = .5
    return thresh


def make_distance_matrix(enrichment_type):
    # Make a distance matrix for no enrichment, positive enrichment, and negative enrichment

    if enrichment_type == "none":
        # Create a 60 x 60 euclidian distance matrix of random values for no enrichment
        rand_mat = np.random.randint(0, 200, size=(60, 60))
        np.fill_diagonal(rand_mat, 0)
        return rand_mat
    elif enrichment_type == "positive":
        # Create positive enrichment distance matrix where 10 cells mostly positive for marker 1
        # are located close in proximity to 10 cells mostly positive for marker 2.
        # Other included cells are not significantly positive for either marker and are located
        # far from the two positive populations.
        dist_mat_pos = np.zeros((80, 80))
        dist_mat_pos[10:20, :10] = 50
        dist_mat_pos[:10, 10:20] = 50
        dist_mat_pos[20:40, :20] = 200
        dist_mat_pos[:20, 20:40] = 200
        dist_mat_pos[40:80, :40] = 300
        dist_mat_pos[:40, 40:80] = 300
        return dist_mat_pos
    elif enrichment_type == "negative":
        # This creates a distance matrix where there are two groups of cells significant for 2 different
        # markers that are not located near each other (not within the dist_lim).
        dist_mat_neg = np.zeros((60, 60))
        dist_mat_neg[20:40, :20] = 300
        dist_mat_neg[:20, 20:40] = 300
        dist_mat_neg[40:50, :40] = 50
        dist_mat_neg[:40, 40:50] = 50
        dist_mat_neg[50:60, :50] = 200
        dist_mat_neg[:50, 50:60] = 200
        return dist_mat_neg


def make_expression_matrix(enrichment_type):
    # Create the expression matrix with cell labels and patient labels for no enrichment,
    # positive enrichment, and negative enrichment.

    # Column names for columns that are not markers (columns to be excluded)
    excluded_colnames = {0: 'cell_size', 1: 'Background', 14: "HH3",
                         23: "summed_channel", 24: "label", 25: "area", 26: "eccentricity",
                         27: "major_axis_length", 28: "minor_axis_length", 29: "perimeter",
                         30: "fov"}

    if enrichment_type == "none":
        all_patient_data = pd.DataFrame(np.zeros((60, 32)))
        # Assigning values to the patient label and cell label columns
        all_patient_data[30] = "Point8"
        all_patient_data[24] = np.arange(len(all_patient_data[1])) + 1
        # We create two populations of 20 cells each, each positive for a different marker (column index 2 and 3)
        all_patient_data.iloc[0:20, 2] = 1
        all_patient_data.iloc[20:40, 3] = 1
        # We assign the two populations of cells 2 different cell phenotypes
        all_patient_data.iloc[0:20, 31] = 1
        all_patient_data.iloc[20:40, 31] = 2
        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data = all_patient_data.rename(excluded_colnames, axis=1)
        return all_patient_data
    elif enrichment_type == "positive":
        all_patient_data_pos = pd.DataFrame(np.zeros((80, 32)))
        # Assigning values to the patient label and cell label columns
        all_patient_data_pos[30] = "Point8"
        all_patient_data_pos[24] = np.arange(len(all_patient_data_pos[1])) + 1
        # We create 8 cells positive for column index 2, and 8 cells positive for column index 3.
        # These are within the dist_lim in dist_mat_pos (positive enrichment distance matrix).
        all_patient_data_pos.iloc[0:8, 2] = 1
        all_patient_data_pos.iloc[10:18, 3] = 1
        # We assign the two populations of cells 2 different cell phenotypes
        all_patient_data_pos.iloc[0:8, 31] = 1
        all_patient_data_pos.iloc[10:18, 31] = 2
        # We create 4 cells in column index 2 and column index 3 that are also positive
        # for their respective markers.
        all_patient_data_pos.iloc[28:32, 2] = 1
        all_patient_data_pos.iloc[32:36, 3] = 1
        # We assign the two populations of cells 2 different cell phenotypes
        all_patient_data_pos.iloc[28:32, 31] = 1
        all_patient_data_pos.iloc[32:36, 31] = 2
        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_pos = all_patient_data_pos.rename(excluded_colnames, axis=1)
        return all_patient_data_pos
    elif enrichment_type == "negative":
        all_patient_data_neg = pd.DataFrame(np.zeros((60, 32)))
        # Assigning values to the patient label and cell label columns
        all_patient_data_neg[30] = "Point8"
        all_patient_data_neg[24] = np.arange(len(all_patient_data_neg[1])) + 1
        # We create two groups of 20 cells positive for marker 1 (in column index 2)
        # and marker 2 (in column index 3) respectively.
        # The two populations are not within the dist_lim in dist_mat_neg (negative enrichment distance matrix)
        all_patient_data_neg.iloc[0:20, 2] = 1
        all_patient_data_neg.iloc[20:40, 3] = 1
        # We assign the two populations of cells 2 different cell phenotypes
        all_patient_data_neg.iloc[0:20, 31] = 1
        all_patient_data_neg.iloc[20:40, 31] = 2
        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_neg = all_patient_data_neg.rename(excluded_colnames, axis=1)
        return all_patient_data_neg


def test_calculate_channel_spatial_enrichment():
    # Test z and p values

    marker_thresholds = make_threshold_mat()

    # Positive enrichment
    all_patient_data_pos = make_expression_matrix("positive")
    dist_mat_pos = make_distance_matrix("positive")
    values, stats = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_pos, marker_thresholds, all_patient_data_pos, bootstrap_num=100)
    # z, muhat, sigmahat, p, h, adj_p, marker_titles = stats[0]
    assert stats.loc["Point8", "p_pos", 2, 3] < .05
    assert stats.loc["Point8", "p_neg", 2, 3] > .05
    assert stats.loc["Point8", "z", 2, 3] > 0
    # Negative enrichment
    all_patient_data_neg = make_expression_matrix("negative")
    dist_mat_neg = make_distance_matrix("negative")
    values, stats = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_neg, marker_thresholds, all_patient_data_neg, bootstrap_num=100)
    # z, muhat, sigmahat, p, h, adj_p, marker_titles = stats[0]
    assert stats.loc["Point8", "p_neg", 2, 3] < .05
    assert stats.loc["Point8", "p_pos", 2, 3] > .05
    assert stats.loc["Point8", "z", 2, 3] < 0
    # No enrichment
    all_patient_data = make_expression_matrix("none")
    dist_mat = make_distance_matrix("none")
    values, stats = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat, marker_thresholds, all_patient_data, bootstrap_num=100)
    # z, muhat, sigmahat, p, h, adj_p, marker_titles = stats[0]
    assert stats.loc["Point8", "p_pos", 2, 3] > .05
    assert stats.loc["Point8", "p_pos", 2, 3] > .05
    assert abs(stats.loc["Point8", "z", 2, 3]) < 2
