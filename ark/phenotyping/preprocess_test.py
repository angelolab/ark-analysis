import os
import pytest

import numpy as np
import pandas as pd
import xarray as xr

import ark.phenotyping.preprocess as preprocess
import ark.utils.misc_utils as misc_utils
import ark.utils.test_utils as test_utils


def test_create_pixel_matrix():
    # must be float64, since that's how Candace's data comes in
    sample_img_xr = test_utils.make_images_xarray(tif_data=None,
                                                  fov_ids=['fov0', 'fov1'],
                                                  channel_names=['chan0', 'chan1'],
                                                  dtype='float64')

    sample_labels = test_utils.make_labels_xarray(label_data=None,
                                                  fov_ids=['fov0', 'fov1'],
                                                  compartment_names=['whole_cell'])

    # pass invalid fov names
    with pytest.raises(ValueError):
        preprocess.create_pixel_matrix(sample_img_xr, sample_labels, fovs=['fov1', 'fov2'])

    # pass invalid channel names
    with pytest.raises(ValueError):
        preprocess.create_pixel_matrix(sample_img_xr, sample_labels, channels=['chan1', 'chan2'])

    # test on all fovs and channels
    flowsom_data = preprocess.create_pixel_matrix(sample_img_xr, sample_labels)

    # assert all fovs and channels are covered
    misc_utils.verify_same_elements(flowsom_fovs=flowsom_data['fov'].unique(),
                                    provided_fovs=['fov0', 'fov1'])
    misc_utils.verify_same_elements(flowsom_chans=flowsom_data.columns.values[:-4],
                                    provided_chans=['chan0', 'chan1'])

    # assert no rows sum to 0
    assert np.all(flowsom_data.loc[:, ['chan0', 'chan1']].sum(axis=1).values != 0)

    # test on a subset of fovs
    flowsom_data = preprocess.create_pixel_matrix(sample_img_xr, sample_labels, fovs=['fov1'])

    misc_utils.verify_same_elements(flowsom_fovs=flowsom_data['fov'].unique(),
                                    provided_fovs=['fov1'])
    misc_utils.verify_same_elements(flowsom_chans=flowsom_data.columns.values[:-4],
                                    provided_chans=['chan0', 'chan1'])

    assert np.all(flowsom_data.loc[:, ['chan0', 'chan1']].sum(axis=1).values != 0)

    # test on a subset of channels
    flowsom_data = preprocess.create_pixel_matrix(sample_img_xr, sample_labels, channels=['chan0'])

    misc_utils.verify_same_elements(flowsom_fovs=flowsom_data['fov'].unique(),
                                    provided_fovs=['fov0', 'fov1'])
    misc_utils.verify_same_elements(flowsom_chans=flowsom_data.columns.values[:-4],
                                    provided_chans=['chan0'])

    # test on a subset of fovs and channels
    flowsom_data = preprocess.create_pixel_matrix(sample_img_xr, sample_labels,
                                                  fovs=['fov1'], channels=['chan0'])

    misc_utils.verify_same_elements(flowsom_fovs=flowsom_data['fov'].unique(),
                                    provided_fovs=['fov1'])
    misc_utils.verify_same_elements(flowsom_chans=flowsom_data.columns.values[:-4],
                                    provided_chans=['chan0'])

    assert np.all(flowsom_data.loc[:, ['chan0', 'chan1']].sum(axis=1).values != 0)
