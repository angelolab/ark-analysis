import numpy as np
from segmentation.utils import spatialanalysis_utils


def test_calc_dist_matrix():
    test_mat = np.zeros((512, 512), dtype="int")
    test_mat[0, 20] = 1
    test_mat[4, 17] = 2

    dist_matrix = spatialanalysis_utils.calc_dist_matrix(test_mat)
    real_mat = np.array([[0, 5], [5, 0]])
    assert np.array_equal(dist_matrix, real_mat)
