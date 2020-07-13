import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import os
from scipy.spatial.distance import cdist
from segmentation.utils import synthetic_spatial_datagen
import importlib
importlib.reload(synthetic_spatial_datagen)


def test_direct_init_dist_matrix():
    # this function tests the functionality of a random initialization of a
    # distance matrix directly.
    # the tests are fairly basic for now but will be expanded on.
    # for now, we just want to ensure two things:
        # matrix symmetry: a property of distance matrices
        # mean AB > mean AC: the mean distance between cells of types A and B
        # is greater than that of types A and C

    # we'll be using the default parameters provided in the functions
    # except for the random seed, which we specify
    sample_dist_mat = synthetic_spatial_datagen.direct_init_dist_matrix(seed=42)

    # assert matrix symmetry
    assert np.allclose(sample_dist_mat, sample_dist_mat.T, rtol=1e-05, atol=1e-08)

    # assert the average of the distance between A and B is smaller
    # than the average of the distance between A and C.
    # this may not eliminate the possibility that the null is proved true
    # but it's definitely a great check that can ensure greater success
    assert sample_dist_mat[:100, 100:200].mean() < sample_dist_mat[:100, 200:].mean()
