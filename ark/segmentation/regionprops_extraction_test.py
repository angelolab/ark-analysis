import copy

import numpy as np
import pandas as pd

from skimage.draw import ellipse
from skimage.measure import moments, regionprops, regionprops_table

import ark.segmentation.regionprops_extraction as regionprops_extraction


def test_major_minor_axis_ratio():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    major_minor_rat = regionprops_extraction.major_minor_axis_ratio(prop_info)
    assert np.round(major_minor_rat, 4) == 1.1524


def test_perim_square_over_area():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    perim_area_rat = regionprops_extraction.perim_square_over_area(prop_info)
    assert np.round(perim_area_rat, 4) == 39.3630


def test_major_axis_equiv_diam_ratio():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    major_diam_rat = regionprops_extraction.major_axis_equiv_diam_ratio(prop_info)
    assert np.round(major_diam_rat, 4) == 1.7664


def test_convex_hull_resid():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    convex_res = regionprops_extraction.convex_hull_resid(prop_info)
    assert np.round(convex_res, 4) == 0.6605


def test_centroid_dif():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    centroid_dist = regionprops_extraction.centroid_dif(prop_info)
    assert np.round(centroid_dist, 4) == 0.3562


def test_num_concavities():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(25, 25, 10, 20, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    # test default cutoffs
    num_concavities = regionprops_extraction.num_concavities(prop_info)
    assert num_concavities == 0

    # define custom thresholds
    kwargs = {
        'small_concavity_minimum': 1,
        'max_compactness': 5,
        'large_concavity_minimum': 10
    }

    # test kwarg-set thresholds
    num_concavities = regionprops_extraction.num_concavities(prop_info, **kwargs)
    assert num_concavities == 2
