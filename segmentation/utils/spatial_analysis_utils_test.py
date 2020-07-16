import numpy as np
import pandas as pd
import xarray as xr
import random
from segmentation.utils import spatial_analysis_utils
from segmentation.utils import spatial_analysis
import importlib
importlib.reload(spatial_analysis)
importlib.reload(spatial_analysis_utils)


def test_calc_dist_matrix():
    test_mat_data = np.zeros((2, 512, 512, 1), dtype="int")
    # Create pythagorean triple to test euclidian distance
    test_mat_data[0, 0, 20] = 1
    test_mat_data[0, 4, 17] = 2
    test_mat_data[1, 5, 25] = 1
    test_mat_data[1, 9, 22] = 2

    coords = [["1", "2"], range(test_mat_data[0].data.shape[0]),
              range(test_mat_data[0].data.shape[1]), ["segmentation_label"]]
    dims = ["fovs", "rows", "cols", "channels"]
    test_mat = xr.DataArray(test_mat_data, coords=coords, dims=dims)

    distance_mat = spatial_analysis_utils.calc_dist_matrix(test_mat)
    real_mat = np.array([[0, 5], [5, 0]])

    assert np.array_equal(distance_mat["1"], real_mat)
    assert np.array_equal(distance_mat["2"], real_mat)


def make_threshold_mat():
    thresh = pd.DataFrame(np.zeros((20, 2)))
    thresh.iloc[:, 1] = .5
    return thresh


def make_example_data_closenum():
    # Creates example data for the creation of the closenum matrix in the below test function

    # Create example all_patient_data cell expression matrix
    all_data = pd.DataFrame(np.zeros((10, 31)))
    # Assigning values to the patient label and cell label columns
    all_data[30] = "Point8"
    all_data[24] = np.arange(len(all_data[1])) + 1

    colnames = {24: "cellLabelInImage", 30: "SampleID", 31: "FlowSOM_ID", 32: "cell_type"}
    all_data = all_data.rename(colnames, axis=1)

    # Create 4 cells positive for marker 1 and 2, 5 cells positive for markers 3 and 4,
    # and 1 cell positive for marker 5
    all_data.iloc[0:4, 2] = 1
    all_data.iloc[0:4, 3] = 1
    all_data.iloc[4:9, 5] = 1
    all_data.iloc[4:9, 6] = 1
    all_data.iloc[9, 7] = 1
    all_data.iloc[9, 8] = 1

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

    return all_data, dist_mat


def test_compute_close_cell_num():
    # Test the closenum function
    all_data, example_dist_mat = make_example_data_closenum()
    example_thresholds = make_threshold_mat()

    # Only include the columns of markers
    fov_channel_data = all_data.drop(all_data.columns[[
        0, 1, 14, 23, 24, 25, 26, 27, 28, 29, 30]], axis=1)

    # List of all markers
    marker_titles = fov_channel_data.columns
    # Length of marker list
    marker_num = len(marker_titles)

    # Subsetting threshold matrix to only include column with threshold values
    thresh_vec = example_thresholds.iloc[0:20, 1]

    example_closenum, m1, m2 = spatial_analysis_utils.compute_close_cell_num(
        dist_mat=example_dist_mat, dist_lim=100, num=marker_num, analysis_type="Channel",
        fov_data=all_data, fov_channel_data=fov_channel_data, thresh_vec=thresh_vec)
    assert (example_closenum[:2, :2] == 16).all()
    assert (example_closenum[3:5, 3:5] == 25).all()
    assert (example_closenum[5:7, 5:7] == 1).all()

    # Now test indexing with cell labels by removing a cell label from the expression matrix but not the
    # distance matrix
    all_data = all_data.drop(3, axis=0)
    # Only include the columns of markers
    fov_channel_data = all_data.drop(all_data.columns[[
        0, 1, 14, 23, 24, 25, 26, 27, 28, 29, 30]], axis=1)
    # List of all markers
    marker_titles = fov_channel_data.columns
    # Length of marker list
    marker_num = len(marker_titles)
    # Subsetting threshold matrix to only include column with threshold values
    thresh_vec = example_thresholds.iloc[0:20, 1]
    example_closenum, m1, m2 = spatial_analysis_utils.compute_close_cell_num(
        dist_mat=example_dist_mat, dist_lim=100, num=marker_num, analysis_type="Channel",
        fov_data=all_data, fov_channel_data=fov_channel_data, thresh_vec=thresh_vec)
    assert (example_closenum[:2, :2] == 9).all()
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
    data_markers, example_distmat = make_example_data_closenum()

    # Generate random inputs to test shape
    marker1_num = [random.randrange(0, 10) for i in range(20)]
    marker2_num = [random.randrange(0, 5) for i in range(400)]
    marker_num = 20

    example_closenumrand = spatial_analysis_utils.compute_close_cell_num_random(
        marker1_num, marker2_num, example_distmat, marker_num, dist_lim=100,
        bootstrap_num=100)

    assert example_closenumrand.shape == (20, 20, 100)


def test_calculate_enrichment_stats():
    # Positive enrichment

    # Generate random closenum matrix
    stats_cnp = np.zeros((20, 20))
    stats_cnp[:, :] = 80

    # Generate random closenumrand matrix, ensuring significant positive enrichment
    stats_cnrp = np.random.randint(1, 40, (20, 20, 100))

    stats_xr_pos = spatial_analysis_utils.calculate_enrichment_stats(stats_cnp, stats_cnrp)

    assert stats_xr_pos.loc["z", 0, 0] > 0
    assert stats_xr_pos.loc["p_pos", 0, 0] < .05

    # Negative enrichment

    # Generate random closenum matrix
    stats_cnn = np.zeros((20, 20))

    # Generate random closenumrand matrix, ensuring significant negative enrichment
    stats_cnrn = np.random.randint(40, 80, (20, 20, 100))

    stats_xr_neg = spatial_analysis_utils.calculate_enrichment_stats(stats_cnn, stats_cnrn)

    assert stats_xr_neg.loc["z", 0, 0] < 0
    assert stats_xr_neg.loc["p_neg", 0, 0] < .05

    # No enrichment

    # Generate random closenum matrix
    stats_cn = np.zeros((20, 20))
    stats_cn[:, :] = 80

    # Generate random closenumrand matrix, ensuring no enrichment
    stats_cnr = np.random.randint(78, 82, (20, 20, 100))

    stats_xr = spatial_analysis_utils.calculate_enrichment_stats(stats_cn, stats_cnr)

    assert abs(stats_xr.loc["z", 0, 0]) < 1
    assert stats_xr.loc["p_neg", 0, 0] > .05
    assert stats_xr.loc["p_pos", 0, 0] > .05
