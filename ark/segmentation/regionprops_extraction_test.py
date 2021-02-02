import copy

import numpy as np
import pandas as pd

from skimage.draw import ellipse
from skimage.measure import moments, regionprops, regionprops_table

import ark.segmentation.regionprops_extraction as regionprops_extraction


def test_centroid_dif():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    centroid_dist = regionprops_extraction.centroid_dif(prop_info)

    assert np.round(centroid_dist, 4) == 0.3562


def test_num_concavities():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    num_concavities = regionprops_extraction.num_concavities(prop_info)

    assert num_concavities == 1
