import numpy as np
import pandas as pd
import random
from segmentation.utils import spatial_analysis_utils
from segmentation.utils import spatial_analysis
import importlib
importlib.reload(spatial_analysis)
importlib.reload(spatial_analysis_utils)


def test_calc_dist_matrix():
    test_mat = np.zeros((1, 512, 512), dtype="int")
    # Create pythagorean triple to test euclidian distance
    test_mat[0, 0, 20] = 1
    test_mat[0, 4, 17] = 2

    dist_matrix_xr = spatial_analysis_utils.calc_dist_matrix(test_mat)
    real_mat = np.array([[0, 5], [5, 0]])
    assert np.array_equal(dist_matrix_xr[0, :, :], real_mat)


# def test_distmat():
#     testtiff = io.imread(
#         "/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/newLmod.tiff")
#     distmat = np.asarray(pd.read_csv(
#         "/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header=None))
#     testmat = spatialanalysis_utils.calc_dist_matrix(testtiff)
#     assert np.allclose(distmat, testmat)


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
        all_patient_data = pd.DataFrame(np.zeros((60, 31)))
        # Assigning values to the patient label and cell label columns
        all_patient_data[30] = "Point8"
        all_patient_data[24] = np.arange(len(all_patient_data[1])) + 1
        # We create two populations of 20 cells each, each positive for a different marker (column index 2 and 3)
        all_patient_data.iloc[0:20, 2] = 1
        all_patient_data.iloc[20:40, 3] = 1
        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data = all_patient_data.rename(excluded_colnames, axis=1)
        return all_patient_data
    elif enrichment_type == "positive":
        all_patient_data_pos = pd.DataFrame(np.zeros((80, 31)))
        # Assigning values to the patient label and cell label columns
        all_patient_data_pos[30] = "Point8"
        all_patient_data_pos[24] = np.arange(len(all_patient_data_pos[1])) + 1
        # We create 8 cells positive for column index 2, and 8 cells positive for column index 3.
        # These are within the dist_lim in dist_mat_pos (positive enrichment distance matrix).
        all_patient_data_pos.iloc[0:8, 2] = 1
        all_patient_data_pos.iloc[10:18, 3] = 1
        # We create 4 cells in column index 2 and column index 3 that are also positive
        # for their respective markers.
        all_patient_data_pos.iloc[28:32, 2] = 1
        all_patient_data_pos.iloc[32:36, 3] = 1
        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_pos = all_patient_data_pos.rename(excluded_colnames, axis=1)
        return all_patient_data_pos
    elif enrichment_type == "negative":
        all_patient_data_neg = pd.DataFrame(np.zeros((60, 31)))
        # Assigning values to the patient label and cell label columns
        all_patient_data_neg[30] = "Point8"
        all_patient_data_neg[24] = np.arange(len(all_patient_data_neg[1])) + 1
        # We create two groups of 20 cells positive for marker 1 (in column index 2)
        # and marker 2 (in column index 3) respectively.
        # The two populations are not within the dist_lim in dist_mat_neg (negative enrichment distance matrix)
        all_patient_data_neg.iloc[0:20, 2] = 1
        all_patient_data_neg.iloc[20:40, 3] = 1
        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_neg = all_patient_data_neg.rename(excluded_colnames, axis=1)
        return all_patient_data_neg


def make_threshold_mat():
    thresh = pd.DataFrame(np.zeros((20, 2)))
    thresh.iloc[:, 1] = .5
    return thresh


def make_example_data_closenum():
    # Creates example data for the creation of the closenum matrix in the below test function

    # Create example all_patient_data cell expression matrix
    all_patient_data = pd.DataFrame(np.zeros((10, 31)))
    # Assigning values to the patient label and cell label columns
    all_patient_data[30] = "Point8"
    all_patient_data[24] = np.arange(len(all_patient_data[1])) + 1
    # Create 4 cells positive for marker 1 and 2, 5 cells positive for markers 3 and 4,
    # and 1 cell positive for marker 5
    all_patient_data.iloc[0:4, 2] = 1
    all_patient_data.iloc[0:4, 3] = 1
    all_patient_data.iloc[4:9, 5] = 1
    all_patient_data.iloc[4:9, 6] = 1
    all_patient_data.iloc[9, 7] = 1
    all_patient_data.iloc[9, 8] = 1

    # Create the distance matrix to test the closenum function
    dist_mat = np.zeros((10, 10))
    np.fill_diagonal(dist_mat, 0)
    # Create distance matrix where cells positive for marker 1 and 2 are within the dist_lim of each other,
    # but not the other groups. This is repeated for cells positive for marker 3 and 4, and for cells positive
    # for marker 5.
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

    return all_patient_data, dist_mat


def test_compute_close_cell_num():
    # Test the closenum function
    example_all_patient_data, example_dist_mat = make_example_data_closenum()
    example_thresholds = make_threshold_mat()

    # Subsets the expression matrix to only have marker columns
    data_markers = example_all_patient_data.drop(example_all_patient_data.columns[[
        0, 1, 14, 23, 24, 25, 26, 27, 28, 29, 30]], axis=1)
    # List of all markers
    marker_titles = data_markers.columns
    # Length of marker list
    marker_num = len(marker_titles)

    # Subsetting threshold matrix to only include column with threshold values
    thresh_vec = example_thresholds.iloc[0:20, 1]

    example_closenum, marker1_num, marker2_num = spatial_analysis.compute_close_cell_num(
        example_all_patient_data, data_markers, thresh_vec, example_dist_mat, marker_num,
        dist_lim=100, cell_label_idx=24)
    assert (example_closenum[:2, :2] == 16).all()
    assert (example_closenum[3:5, 3:5] == 25).all()
    assert (example_closenum[5:7, 5:7] == 1).all()

    # Now test with Erin's output
    # cell_array = pd.read_csv(
    #     "/Users/jaiveersingh/Downloads/SpatialEnrichm"
    #     "ent/granA_cellpheno_CS-asinh-norm_revised.csv")
    # marker_thresholds = pd.read_csv(
    #     "/Users/jaiveersingh/Downloads/SpatialEnrichment/markerThresholds.csv")
    # dist_matrix = np.asarray(pd.read_csv(
    #     "/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header=None))
    # closenum, closenumRand, z, muhat, sigmahat, p, h, adj_p, markertitles = \
    #     spatialorgmarkerexpression_utils.calculate_channel_spatial_enrichment(
    #         dist_matrix, marker_thresholds, cell_array)
    # real_closenum = np.asarray(pd.read_csv(
    #     "/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/closeNum.csv"))
    # assert np.array_equal(closenum, real_closenum)


def test_compute_close_cell_num_random():
    example_all_patient_data, example_distmat = make_example_data_closenum()

    # Generate random inputs to test shape
    marker1_num = [random.randrange(0, 10) for i in range(20)]
    marker2_num = [random.randrange(0, 5) for i in range(400)]
    marker_num = 20

    example_closenumrand = spatial_analysis.compute_close_cell_num_random(
        marker1_num, marker2_num, example_all_patient_data, example_distmat, marker_num, dist_lim=100,
        cell_label_idx=24, bootstrap_num=100)

    assert example_closenumrand.shape == (20, 20, 100)


def test_calculate_enrichment_stats():
    # Positive enrichment

    # Generate random closenum matrix
    stats_cnp = np.zeros((20, 20))
    stats_cnp[:, :] = 80

    # Generate random closenumrand matrix, ensuring significant positive enrichment
    stats_cnrp = np.random.randint(1, 40, (20, 20, 100))

    stats_xr_pos = spatial_analysis.calculate_enrichment_stats(stats_cnp, stats_cnrp)

    assert stats_xr_pos.loc["z", 0, 0] > 0
    assert stats_xr_pos.loc["p_pos", 0, 0] < .05

    # Negative enrichment

    # Generate random closenum matrix
    stats_cnn = np.zeros((20, 20))

    # Generate random closenumrand matrix, ensuring significant negative enrichment
    stats_cnrn = np.random.randint(40, 80, (20, 20, 100))

    stats_xr_neg = spatial_analysis.calculate_enrichment_stats(stats_cnn, stats_cnrn)

    assert stats_xr_neg.loc["z", 0, 0] < 0
    assert stats_xr_neg.loc["p_neg", 0, 0] < .05

    # No enrichment

    # Generate random closenum matrix
    stats_cn = np.zeros((20, 20))
    stats_cn[:, :] = 80

    # Generate random closenumrand matrix, ensuring no enrichment
    stats_cnr = np.random.randint(78, 82, (20, 20, 100))

    stats_xr = spatial_analysis.calculate_enrichment_stats(stats_cn, stats_cnr)

    assert abs(stats_xr.loc["z", 0, 0]) < 1
    assert stats_xr.loc["p_neg", 0, 0] > .05
    assert stats_xr.loc["p_pos", 0, 0] > .05


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
