import numpy as np
from segmentation.utils import Spatial_Analysis

def test_calc_dist_matrix():

    test_mat = np.zeros((512, 512), dtype="int")

    test_mat[40, 20] = 1
    test_mat[44, 17] = 2

    dist_matrix = Spatial_Analysis.calc_dist_matrix(test_mat)
    return dist_matrix

print(test_calc_dist_matrix())

