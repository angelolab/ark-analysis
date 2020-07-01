import sys
import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import os
from scipy.spatial.distance import cdist

sys.path.append('..')
from segmentation.utils import synthetic_spatial_datagen


# random seed
RSEED = 42


def test_direct_init_dist_matrix():
    # we'll be using the default parameters provided in the functions
    # except for the random seed, which we specify
    sample_dist_matrix = synthetic_spatial_datagen.direct_init_dist_matrix(seed=RSEED)

    # assert matrix symmetry
    assert np.allclose(sample_dist_mat, sample_dist_mat.T, rtol=1e-05, atol=1e-08)

    # assert the average of the distance between A and B is greater 
    # than the average of the distance between A and C.
    # this may not eliminate the possibility that the null is proved true
    # but it's definitely a great check that can ensure greater success
    assert sample_dist_matrix[:100, 100:200].mean() > sample_dist_matrix[:100, 200:]

def test_point_init_dist_matrix():
    # we'll be using the default parameters provided in the functions
    # except for the random seed, which we specify
    sample_dist_matrix = synthetic_spatial_datagen.test_point_init_dist_matrix(seed=RSEED)

    # assert matrix symmetry
    assert np.allclose(sample_dist_mat, sample_dist_mat.T, rtol=1e-05, atol=1e-08)

    # assert the average of the distance between A and B is greater 
    # than the average of the distance between A and C.
    # this may not eliminate the possibility that the null is proved true
    # but it's definitely a great check that can ensure greater success
    assert sample_dist_matrix[:100, 100:200].mean() > sample_dist_matrix[:100, 200:]