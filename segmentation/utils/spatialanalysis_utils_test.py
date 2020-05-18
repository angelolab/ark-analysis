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


"""
def test_distmat():
    testtiff = io.imread(
        "/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/newLmod.tiff")
    distmat = np.asarray(pd.read_csv(
        "/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header=None))
    testmat = spatialanalysis_utils.calc_dist_matrix(testtiff)
    assert np.allclose(distmat, testmat)
"""


def make_distance_matrix(typeofenfrichment):
    if(typeofenfrichment == "none"):
        randmat = np.random.randint(0, 200, size=(60, 60))
        np.fill_diagonal(randmat, 0)
        return randmat
    elif(typeofenfrichment == "positive"):
        distmatp = np.zeros((80, 80))
        distmatp[10:20, :10] = 50
        distmatp[:10, 10:20] = 50
        distmatp[20:40, :20] = 200
        distmatp[:20, 20:40] = 200
        distmatp[40:80, :40] = 300
        distmatp[:40, 40:80] = 300
        return distmatp
    elif(typeofenfrichment == "negative"):
        distmatn = np.zeros((60, 60))
        distmatn[20:40, :20] = 300
        distmatn[:20, 20:40] = 300
        distmatn[40:50, :40] = 50
        distmatn[:40, 40:50] = 50
        distmatn[50:60, :50] = 200
        distmatn[:50, 50:60] = 200
        return distmatn


def make_expression_matrix(typeofencrichment):
    if(typeofencrichment == "none"):
        cellarray = pd.DataFrame(np.zeros((60, 53)))
        cellarray[0] = 6
        cellarray[1] = np.arange(len(cellarray[1])) + 1
        cellarray.iloc[0:20, 7] = 1
        cellarray.iloc[20:40, 8] = 1
        return cellarray
    elif(typeofencrichment == "positive"):
        cellarrayp = pd.DataFrame(np.zeros((80, 53)))
        cellarrayp[0] = 6
        cellarrayp[1] = np.arange(len(cellarrayp[1])) + 1
        cellarrayp.iloc[0:8, 7] = 1
        cellarrayp.iloc[28:30, 7] = 1
        cellarrayp.iloc[38:40, 7] = 1
        cellarrayp.iloc[10:18, 8] = 1
        cellarrayp.iloc[27, 8] = 1
        cellarrayp.iloc[30, 8] = 1
        cellarrayp.iloc[36:38, 8] = 1
        return cellarrayp
    elif(typeofencrichment == "negative"):
        cellarrayn = pd.DataFrame(np.zeros((60, 53)))
        cellarrayn[0] = 6
        cellarrayn[1] = np.arange(len(cellarrayn[1])) + 1
        cellarrayn.iloc[0:20, 7] = 1
        cellarrayn.iloc[20:40, 8] = 1
        return cellarrayn


def make_threshold_mat():
    thresh = pd.DataFrame(np.zeros((37, 2)))
    thresh.iloc[:, 1] = .5
    return thresh


def test_spatial_analysis():

    """
    # test the closenum function
    test_cellarray = pd.read_csv(
        "/Users/jaiveersingh/Desktop/tests/celllabels.csv")
    test_thresholds = pd.read_csv(
        "/Users/jaiveersingh/Desktop/tests/thresholds.csv")
    test_distmat = np.asarray(pd.read_csv(
        "/Users/jaiveersingh/Desktop/tests/distmat.csv", header=None))
    test_closeNum, closeNumRand, z, muhat, sigmahat, p, h, adj_p, \
        markerTitles = spatialorgmarkerexpression_utils.spatial_analysis(
            test_distmat, test_thresholds, test_cellarray)
    assert (test_closeNum[:2, :2] == 16).all()
    assert (test_closeNum[2:4, 2:4] == 25).all()
    assert (test_closeNum[4:6, 4:6] == 1).all()

    # Now test with Erin's output
    cell_array = pd.read_csv(
        "/Users/jaiveersingh/Downloads/SpatialEnrichm"
        "ent/granA_cellpheno_CS-asinh-norm_revised.csv")
    marker_thresholds = pd.read_csv(
        "/Users/jaiveersingh/Downloads/SpatialEnrichment/markerThresholds.csv")
    dist_matrix = np.asarray(pd.read_csv(
        "/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header=None))
    closenum, closenumRand, z, muhat, sigmahat, p, h, adj_p, markertitles = \
        spatialorgmarkerexpression_utils.spatial_analysis(
            dist_matrix, marker_thresholds, cell_array)
    real_closenum = np.asarray(pd.read_csv(
        "/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/closeNum.csv"))
    assert np.array_equal(closenum, real_closenum)
    """

    # test z and p values
    marker_thresholds = make_threshold_mat()
    # positive enrichment
    cellarrayp = make_expression_matrix("positive")
    distmatp = make_distance_matrix("positive")
    closenum, closenumrand, zp, muhat, sigmahat, pp, h, adj_p, markertitles = \
        spatialorgmarkerexpression_utils.spatial_analysis(
            distmatp, marker_thresholds, cellarrayp)
    assert pp[0, 1, 0] < .05
    assert pp[0, 1, 1] > .05
    assert zp[0, 1] > 0
    # negative enrichment
    cellarrayn = make_expression_matrix("negative")
    distmatn = make_distance_matrix("negative")
    closenum, closenumrand, zn, muhatn, sigmahatn, pn, h, adj_p, markertitles = \
        spatialorgmarkerexpression_utils.spatial_analysis(
            distmatn, marker_thresholds, cellarrayn)
    assert pn[0, 1, 1] < .05
    assert pn[0, 1, 0] > .05
    assert zn[0, 1] < 0
    # no enrichment
    cellarray = make_expression_matrix("none")
    distmat = make_distance_matrix("none")
    closenum, closenumrand, z, muhat, sigmahat, p, h, adj_p, markertitles = \
        spatialorgmarkerexpression_utils.spatial_analysis(
            distmat, marker_thresholds, cellarray)
    assert p[0, 1, 0] > .05
    assert p[0, 1, 1] < .05
    assert abs(z[0, 1]) < 1
