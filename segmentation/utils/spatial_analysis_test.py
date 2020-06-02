import numpy as np
import pandas as pd
import xarray as xr
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
        rand_mat = np.random.randint(0, 200, size=(1, 60, 60))
        np.fill_diagonal(rand_mat[0, :, :], 0)
        coords = [range(rand_mat.shape[0]), range(rand_mat[0].data.shape[0]), range(rand_mat[0].data.shape[1])]
        dims = ["points", "rows", "cols"]
        rand_mat = xr.DataArray(rand_mat, coords=coords, dims=dims)
        return rand_mat
    elif enrichment_type == "positive":
        # Create positive enrichment distance matrix where 10 cells mostly positive for marker 1
        # are located close in proximity to 10 cells mostly positive for marker 2.
        # Other included cells are not significantly positive for either marker and are located
        # far from the two positive populations.
        dist_mat_pos = np.zeros((1, 80, 80))
        dist_mat_pos[0, 10:20, :10] = 50
        dist_mat_pos[0, :10, 10:20] = 50
        dist_mat_pos[0, 20:40, :20] = 200
        dist_mat_pos[0, :20, 20:40] = 200
        dist_mat_pos[0, 40:80, :40] = 300
        dist_mat_pos[0, :40, 40:80] = 300
        coords = [range(dist_mat_pos.shape[0]), range(dist_mat_pos[0].data.shape[0]), range(dist_mat_pos[0].data.shape[1])]
        dims = ["points", "rows", "cols"]
        dist_mat_pos = xr.DataArray(dist_mat_pos, coords=coords, dims=dims)
        return dist_mat_pos
    elif enrichment_type == "negative":
        # This creates a distance matrix where there are two groups of cells significant for 2 different
        # markers that are not located near each other (not within the dist_lim).
        dist_mat_neg = np.zeros((1, 60, 60))
        dist_mat_neg[0, 20:40, :20] = 300
        dist_mat_neg[0, :20, 20:40] = 300
        dist_mat_neg[0, 40:50, :40] = 50
        dist_mat_neg[0, :40, 40:50] = 50
        dist_mat_neg[0, 50:60, :50] = 200
        dist_mat_neg[0, :50, 50:60] = 200
        coords = [range(dist_mat_neg.shape[0]), range(dist_mat_neg[0].data.shape[0]), range(dist_mat_neg[0].data.shape[1])]
        dims = ["points", "rows", "cols"]
        dist_mat_neg = xr.DataArray(dist_mat_neg, coords=coords, dims=dims)
        return dist_mat_neg


def make_expression_matrix(enrichment_type):
    # Create the expression matrix with cell labels and patient labels for no enrichment,
    # positive enrichment, and negative enrichment.

    # Column names for columns that are not markers (columns to be excluded)
    excluded_colnames = {0: 'cell_size', 1: 'Background', 14: "HH3",
                         23: "summed_channel", 24: "cellLabelInImage", 25: "area", 26: "eccentricity",
                         27: "major_axis_length", 28: "minor_axis_length", 29: "perimeter",
                         30: "SampleID", 31: "FlowSOM_ID", 32: "cell_type"}

    if enrichment_type == "none":
        all_data = pd.DataFrame(np.zeros((60, 33)))
        # Assigning values to the patient label and cell label columns
        all_data[30] = "Point8"
        all_data[24] = np.arange(len(all_data[1])) + 1
        # We create two populations of 20 cells each, each positive for a different marker (column index 2 and 3)
        all_data.iloc[0:20, 2] = 1
        all_data.iloc[20:40, 3] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data.iloc[0:20, 31] = 1
        all_data.iloc[0:20, 32] = "Pheno1"
        all_data.iloc[20:40, 31] = 2
        all_data.iloc[20:40, 32] = "Pheno2"
        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data = all_data.rename(excluded_colnames, axis=1)
        return all_patient_data
    elif enrichment_type == "positive":
        all_data_pos = pd.DataFrame(np.zeros((80, 33)))
        # Assigning values to the patient label and cell label columns
        all_data_pos[30] = "Point8"
        all_data_pos[24] = np.arange(len(all_data_pos[1])) + 1
        # We create 8 cells positive for column index 2, and 8 cells positive for column index 3.
        # These are within the dist_lim in dist_mat_pos (positive enrichment distance matrix).
        all_data_pos.iloc[0:8, 2] = 1
        all_data_pos.iloc[10:18, 3] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data_pos.iloc[0:8, 31] = 1
        all_data_pos.iloc[0:8, 32] = "Pheno1"
        all_data_pos.iloc[10:18, 31] = 2
        all_data_pos.iloc[10:18, 32] = "Pheno2"
        # We create 4 cells in column index 2 and column index 3 that are also positive
        # for their respective markers.
        all_data_pos.iloc[28:32, 2] = 1
        all_data_pos.iloc[32:36, 3] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data_pos.iloc[28:32, 31] = 1
        all_data_pos.iloc[28:32, 32] = "Pheno1"
        all_data_pos.iloc[32:36, 31] = 2
        all_data_pos.iloc[32:36, 32] = "Pheno2"
        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_pos = all_data_pos.rename(excluded_colnames, axis=1)
        return all_patient_data_pos
    elif enrichment_type == "negative":
        all_data_neg = pd.DataFrame(np.zeros((60, 33)))
        # Assigning values to the patient label and cell label columns
        all_data_neg[30] = "Point8"
        all_data_neg[24] = np.arange(len(all_data_neg[1])) + 1
        # We create two groups of 20 cells positive for marker 1 (in column index 2)
        # and marker 2 (in column index 3) respectively.
        # The two populations are not within the dist_lim in dist_mat_neg (negative enrichment distance matrix)
        all_data_neg.iloc[0:20, 2] = 1
        all_data_neg.iloc[20:40, 3] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data_neg.iloc[0:20, 31] = 1
        all_data_neg.iloc[0:20, 32] = "Pheno1"
        all_data_neg.iloc[20:40, 31] = 2
        all_data_neg.iloc[20:40, 32] = "Pheno2"
        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_neg = all_data_neg.rename(excluded_colnames, axis=1)
        return all_patient_data_neg


def test_calculate_channel_spatial_enrichment():

    excluded_colnames = ["cell_size", "Background", "HH3",
                         "summed_channel", "cellLabelInImage", "area",
                         "eccentricity", "major_axis_length", "minor_axis_length",
                         "perimeter", "SampleID", "FlowSOM_ID", "cell_type"]

    # Test z and p values

    marker_thresholds = make_threshold_mat()

    # Positive enrichment
    all_data_pos = make_expression_matrix("positive")
    dist_mat_pos = make_distance_matrix("positive")
    values, stats = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_pos, marker_thresholds, all_data_pos,
            excluded_colnames=excluded_colnames, bootstrap_num=100)
    assert stats.loc["Point8", "p_pos", 2, 3] < .05
    assert stats.loc["Point8", "p_neg", 2, 3] > .05
    assert stats.loc["Point8", "z", 2, 3] > 0
    # Negative enrichment
    all_data_neg = make_expression_matrix("negative")
    dist_mat_neg = make_distance_matrix("negative")
    values, stats = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_neg, marker_thresholds, all_data_neg,
            excluded_colnames=excluded_colnames, bootstrap_num=100)
    assert stats.loc["Point8", "p_neg", 2, 3] < .05
    assert stats.loc["Point8", "p_pos", 2, 3] > .05
    assert stats.loc["Point8", "z", 2, 3] < 0
    # No enrichment
    all_data = make_expression_matrix("none")
    dist_mat = make_distance_matrix("none")
    values, stats = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat, marker_thresholds, all_data,
            excluded_colnames=excluded_colnames, bootstrap_num=100)
    assert stats.loc["Point8", "p_pos", 2, 3] > .05
    assert stats.loc["Point8", "p_pos", 2, 3] > .05
    assert abs(stats.loc["Point8", "z", 2, 3]) < 2


def test_calculate_cluster_spatial_enrichment():
    # Test z and p values

    # Positive enrichment
    all_data_pos = make_expression_matrix("positive")
    dist_mat_pos = make_distance_matrix("positive")
    values, stats = \
        spatial_analysis.calculate_cluster_spatial_enrichment(
            all_data_pos, dist_mat_pos,
            bootstrap_num=100, dist_lim=100)
    assert stats.loc["Point8", "p_pos", "Pheno1", "Pheno2"] < .05
    assert stats.loc["Point8", "p_neg", "Pheno1", "Pheno2"] > .05
    assert stats.loc["Point8", "z", "Pheno1", "Pheno2"] > 0
    # Negative enrichment
    all_data_neg = make_expression_matrix("negative")
    dist_mat_neg = make_distance_matrix("negative")
    values, stats = \
        spatial_analysis.calculate_cluster_spatial_enrichment(
            all_data_neg, dist_mat_neg,
            bootstrap_num=100, dist_lim=100)
    assert stats.loc["Point8", "p_neg", "Pheno1", "Pheno2"] < .05
    assert stats.loc["Point8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert stats.loc["Point8", "z", "Pheno1", "Pheno2"] < 0
    # No enrichment
    all_data = make_expression_matrix("none")
    dist_mat = make_distance_matrix("none")
    values, stats = \
        spatial_analysis.calculate_cluster_spatial_enrichment(
            all_data, dist_mat,
            bootstrap_num=100, dist_lim=100)
    assert stats.loc["Point8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert stats.loc["Point8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert abs(stats.loc["Point8", "z", "Pheno1", "Pheno2"]) < 2
