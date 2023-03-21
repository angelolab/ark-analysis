import os

import numpy as np
import pandas as pd
import pytest
import skimage.io as io
from alpineer import image_utils, test_utils

from ark.phenotyping import post_cluster_utils


def test_plot_hist_thresholds():
    pops = np.repeat(['pop1', 'pop2', 'pop3'], 5)
    marker_1 = np.random.rand(len(pops))

    cell_table = pd.DataFrame({'cell_meta_cluster': pops, 'marker_1': marker_1})

    # populations argument must be a list, but`make_iterable` should convert a `str`
    # argument to `List[str]`
    post_cluster_utils.plot_hist_thresholds(cell_table=cell_table, populations='pop1',
                                            marker='marker_1')

    # populations argument must contain entries from cell_table
    with pytest.raises(ValueError, match='Invalid population'):
        post_cluster_utils.plot_hist_thresholds(cell_table=cell_table,
                                                populations=['pop1', 'pop4'],
                                                marker='marker_1')

    # marker argument must be a column in cell_table
    with pytest.raises(ValueError, match='Could not find'):
        post_cluster_utils.plot_hist_thresholds(cell_table=cell_table,
                                                populations=['pop1', 'pop2'],
                                                marker='marker_2')

    # runs without errors
    post_cluster_utils.plot_hist_thresholds(cell_table=cell_table,
                                            populations=['pop1', 'pop2'],
                                            marker='marker_1')


def test_create_mantis_project(tmp_path):
    # create necessary directories
    seg_dir = os.path.join(tmp_path, 'seg')
    os.makedirs(seg_dir)

    image_dir = os.path.join(tmp_path, 'images')
    os.makedirs((image_dir))

    mantis_dir = os.path.join(tmp_path, 'mantis')
    os.makedirs((mantis_dir))

    mask_dir = os.path.join(tmp_path, 'mask')

    # create images
    fovs, channels = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=4,
                                                   use_delimiter=False, return_imgs=False)
    test_utils._write_tifs(image_dir, fovs, channels, (10, 10), '', False, int)

    # create random segmentation masks
    for fov in fovs:
        data = np.random.randint(0, 5, 100).reshape(10, 10)
        image_utils.save_image(os.path.join(seg_dir, fov + '_whole_cell_test.tiff'), data)

    # create cell table with two clusters
    cell_label = np.tile(np.arange(1, 5), len(fovs))
    cell_clusters = np.tile(['cluster1', 'cluster2'], 6)
    fov_list = np.repeat(fovs, 4)

    cell_table = pd.DataFrame({'fov': fov_list, 'label': cell_label,
                               'cell_meta_cluster': cell_clusters})

    post_cluster_utils.create_mantis_project(cell_table=cell_table, fovs=fovs,
                                             seg_dir=seg_dir, pop_col='cell_meta_cluster',
                                             mask_dir=mask_dir, image_dir=image_dir,
                                             mantis_dir=mantis_dir,
                                             seg_suffix_name="_whole_cell_test.tiff",
                                             cluster_type='cell')

    # make sure that the mask found in each mantis directory is correct
    for fov in fovs:
        # mask should only include 0, 1, and 2 for background, population_1, and population_2
        mask = io.imread(os.path.join(mask_dir, fov + '_cell_mask.tiff'))
        assert set(np.unique(mask)) == set([0, 1, 2])

        # mask should be non-zero in the same places as original
        seg = io.imread(os.path.join(seg_dir, fov + '_whole_cell_test.tiff'))
        assert np.array_equal(mask > 0, seg > 0)
