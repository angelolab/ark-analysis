import numpy as np
import pandas as pd
from skimage import io
from segmentation.utils import spatialanalysis_utils


def test_calc_dist_matrix():
    test_mat = np.zeros((512, 512), dtype="int")
    test_mat[0, 20] = 1
    test_mat[4, 17] = 2

    dist_matrix = spatialanalysis_utils.calc_dist_matrix(test_mat)
    real_mat = np.array([[0, 5], [5, 0]])
    assert np.array_equal(dist_matrix, real_mat)


def test_distmat():
    testtiff = io.imread("/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/newLmod.tiff")
    distMat = np.asarray(pd.read_csv("/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header = None))
    testMat = spatialanalysis_utils.calc_dist_matrix(testtiff)
    assert np.allclose(distMat, testMat)