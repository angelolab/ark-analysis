import numpy as np
import pandas as pd
from skimage import io
from segmentation.utils import spatialanalysis_utils
from segmentation.utils import spatialorgmarkerexpression_utils


def test_calc_dist_matrix():
    test_mat = np.zeros((512, 512), dtype="int")
    test_mat[0, 20] = 1
    test_mat[4, 17] = 2

    dist_matrix = spatialanalysis_utils.calc_dist_matrix(test_mat)
    real_mat = np.array([[0, 5], [5, 0]])
    assert np.array_equal(dist_matrix, real_mat)


def test_distmat():
    testtiff = io.imread(
        "/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/newLmod.tiff")
    distMat = np.asarray(pd.read_csv(
        "/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header=None))
    testMat = spatialanalysis_utils.calc_dist_matrix(testtiff)
    assert np.allclose(distMat, testMat)


def test_spatial_analysis():
    # test the closenum function
    test_cellarray = pd.read_csv(
        "/Users/jaiveersingh/Desktop/tests/celllabels.csv")
    test_thresholds = pd.read_csv(
        "/Users/jaiveersingh/Desktop/tests/thresholds.csv")
    test_distmat = np.asarray(pd.read_csv(
        "/Users/jaiveersingh/Desktop/tests/distmat.csv", header=None))
    test_closeNum = spatialorgmarkerexpression_utils.spatial_analysis(
        test_distmat, test_thresholds, test_cellarray)
    assert test_closeNum[0, 0] == 16
    assert test_closeNum[0, 1] == 16
    assert test_closeNum[1, 0] == 16
    assert test_closeNum[1, 1] == 16
    assert test_closeNum[2, 2] == 25
    assert test_closeNum[2, 3] == 25
    assert test_closeNum[3, 2] == 25
    assert test_closeNum[3, 3] == 25
    assert test_closeNum[4, 4] == 1
    assert test_closeNum[5, 4] == 1
    assert test_closeNum[4, 5] == 1
    assert test_closeNum[5, 5] == 1

    # Now test with Erin's output
    cell_array = pd.read_csv(
        "/Users/jaiveersingh/Downloads/SpatialEnrichm"
        "ent/granA_cellpheno_CS-asinh-norm_revised.csv")
    marker_thresholds = pd.read_csv(
        "/Users/jaiveersingh/Downloads/SpatialEnrichment/markerThresholds.csv")
    dist_matrix = np.asarray(pd.read_csv(
        "/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header=None))
    closeNum = spatialorgmarkerexpression_utils.spatial_analysis(
        dist_matrix, marker_thresholds, cell_array)
    real_closeNum = np.asarray(pd.read_csv(
        "/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/closeNum.csv"))
    assert np.array_equal(closeNum, real_closeNum)

    # test z and p values
    marker_thresholds = pd.read_csv(
        "/Users/jaiveersingh/Downloads/SpatialEnrichment/markerThresholds.csv")
    # positive enrichment
    cellArrayP = spatialanalysis_utils.make_expression_matrix("positive")
    distMatP = spatialanalysis_utils.make_distance_matrix("positive")
    closeNum, closeNumRand, z, muhat, sigmahat, p, h, adj_p, markerTitles = \
        spatialorgmarkerexpression_utils.spatial_analysis(
            distMatP, marker_thresholds, cellArrayP)
    assert p[0, 1, 0] < .05
    assert z[0, 1] > 0
    # negative enrichment
    cellArrayN = spatialanalysis_utils.make_expression_matrix("negative")
    distMatN = spatialanalysis_utils.make_distance_matrix("negative")
    closeNum, closeNumRand, z, muhat, sigmahat, p, h, adj_p, markerTitles = \
        spatialorgmarkerexpression_utils.spatial_analysis(
            distMatN, marker_thresholds, cellArrayN)
    assert p[0, 1, 1] < .05
    assert z[0, 1] < 0
    # no enrichment
    cellArray = spatialanalysis_utils.make_expression_matrix("none")
    distMat = spatialanalysis_utils.make_distance_matrix("none")
    closeNum, closeNumRand, z, muhat, sigmahat, p, h, adj_p, markerTitles = \
        spatialorgmarkerexpression_utils.spatial_analysis(
            distMat, marker_thresholds, cellArray)
    assert p[0, 1, 0] > .05
    assert abs(z[0, 1]) < 1
