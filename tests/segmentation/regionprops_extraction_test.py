import copy

import numpy as np
import xarray as xr
from skimage.draw import ellipse
from skimage.measure import regionprops

import ark.segmentation.regionprops_extraction as regionprops_extraction


def test_major_minor_axis_ratio():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    major_minor_rat = regionprops_extraction.major_minor_axis_ratio(prop_info)
    desired_value = 1.1524
    np.testing.assert_allclose(actual=major_minor_rat, desired=desired_value, rtol=0.1, atol=1e-5)

    class Regionprop(object):
        pass

    prop_info = Regionprop()
    prop_info.major_axis_length = 10
    prop_info.minor_axis_length = 0

    nan_val = regionprops_extraction.major_minor_axis_ratio(prop_info)
    assert np.isnan(nan_val)


def test_perim_square_over_area():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    perim_area_rat = regionprops_extraction.perim_square_over_area(prop_info)
    desired_value = 39.3630
    np.testing.assert_allclose(actual=perim_area_rat, desired=desired_value, rtol=0.1, atol=1e-5)


def test_major_axis_equiv_diam_ratio():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    major_diam_rat = regionprops_extraction.major_axis_equiv_diam_ratio(prop_info)
    desired_value = 1.7664
    np.testing.assert_allclose(actual=major_diam_rat, desired=desired_value, rtol=0.1, atol=1e-5)


def test_convex_hull_resid():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    convex_res = regionprops_extraction.convex_hull_resid(prop_info)
    desired_value = 0.6605
    np.testing.assert_allclose(actual=convex_res, desired=desired_value, rtol=0.1, atol=1e-5)


def test_nc_ratio():
    # create a sample marker count matrix with 2 compartments, 3 cell ids, and 3 features
    sample_marker_counts = np.zeros((2, 3, 3))

    # cell 0: no nucleus
    sample_marker_counts[0, 0, 1] = 5

    # cell 1: equal whole cell and nuclear area
    sample_marker_counts[0, 1, 1] = 10
    sample_marker_counts[1, 1, 1] = 10

    # cell 2: different whole cell and nuclear area
    sample_marker_counts[0, 2, 1] = 10
    sample_marker_counts[1, 2, 1] = 5

    # write marker_counts to xarray
    sample_marker_counts = xr.DataArray(copy.copy(sample_marker_counts),
                                        coords=[['whole_cell', 'nuclear'],
                                                [0, 1, 2],
                                                ['feat_1', 'area', 'nc_ratio']],
                                        dims=['compartments', 'cell_id', 'features'])

    sample_marker_counts = regionprops_extraction.nc_ratio(sample_marker_counts)

    # testing cell 0
    assert sample_marker_counts.loc['whole_cell', 0, 'nc_ratio'] == 0
    assert sample_marker_counts.loc['nuclear', 0, 'nc_ratio'] == 0

    # testing cell 1
    assert sample_marker_counts.loc['whole_cell', 1, 'nc_ratio'] == 1
    assert sample_marker_counts.loc['nuclear', 1, 'nc_ratio'] == 1

    # testing cell 2
    assert sample_marker_counts.loc['whole_cell', 2, 'nc_ratio'] == 0.5
    assert sample_marker_counts.loc['nuclear', 2, 'nc_ratio'] == 0.5


def test_centroid_dif():
    sample_arr = np.zeros((50, 50)).astype(int)

    ellipse_x, ellipse_y = ellipse(10, 10, 15, 15, rotation=35)
    sample_arr[ellipse_x, ellipse_y] = 1

    prop_info = regionprops(sample_arr)[0]

    centroid_dist = regionprops_extraction.centroid_dif(prop_info)
    desired_value = 0.3562
    np.testing.assert_allclose(actual=centroid_dist, desired=desired_value, rtol=0.1, atol=1e-5)


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
