import os
import pytest
import tempfile

import numpy as np
import pandas as pd
import skimage.io as io

from ark import settings
from test_utils import make_cell_table
from alpineer import image_utils, test_utils, misc_utils
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
    seg_dir = os.path.join(tmp_path, "seg")
    os.makedirs(seg_dir)

    image_dir = os.path.join(tmp_path, "images")
    os.makedirs((image_dir))

    mantis_dir = os.path.join(tmp_path, "mantis")
    os.makedirs((mantis_dir))

    mask_dir = os.path.join(tmp_path, "mask")

    # create images
    fovs, channels = test_utils.gen_fov_chan_names(
        num_fovs=3, num_chans=4, use_delimiter=False, return_imgs=False
    )
    test_utils._write_tifs(image_dir, fovs, channels, (10, 10), "", False, int)

    # create random segmentation masks
    for fov in fovs:
        data = np.random.randint(0, 5, 100).reshape(10, 10)
        image_utils.save_image(
            os.path.join(seg_dir, fov + "_whole_cell_test.tiff"), data
        )

    # create cell table with two clusters
    cell_label = np.tile(np.arange(1, 5), len(fovs))
    cell_clusters = np.tile(["cluster1", "cluster2"], 6)
    fov_list = np.repeat(fovs, 4)

    cell_table = pd.DataFrame(
        {"fov": fov_list, "label": cell_label, "cell_meta_cluster": cell_clusters}
    )

    post_cluster_utils.create_mantis_project(
        cell_table=cell_table,
        fovs=fovs,
        seg_dir=seg_dir,
        mask_dir=mask_dir,
        image_dir=image_dir,
        mantis_dir=mantis_dir,
        pop_col="cell_meta_cluster",
        fov_col=settings.FOV_ID,
        label_col=settings.CELL_LABEL,
        seg_suffix_name="_whole_cell_test.tiff",
    )

    # make sure that the mask found in each mantis directory is correct
    for fov in fovs:
        # mask should only include 0, 1, and 2 for background, population_1, and population_2
        mask = io.imread(os.path.join(mask_dir, fov + "_post_clustering_cell_mask.tiff"))
        assert set(np.unique(mask)) == set([0, 1, 2])

        # mask should be non-zero in the same places as original
        seg = io.imread(os.path.join(seg_dir, fov + "_whole_cell_test.tiff"))
        assert np.array_equal(mask > 0, seg > 0)


def test_generate_new_cluster_resolution():
    with tempfile.TemporaryDirectory() as temp_dir:
        cell_table = make_cell_table(n_cells=20, n_markers=0)
        cluster_assignments = {'AB': ['A', 'B'], 'C': ['C']}
        new_path = os.path.join(temp_dir, 'new_table.csv')

        # generate and save a new cell table with new cell cluster resolution
        post_cluster_utils.generate_new_cluster_resolution(
            cell_table, cluster_col=settings.CELL_TYPE, new_cluster_col="new_clusters",
            cluster_mapping=cluster_assignments, save_path=new_path)

        new_table = pd.read_csv(new_path)

        # check new column exists
        assert "new_clusters" in new_table.columns

        # check for new cell cluster names
        assert misc_utils.verify_same_elements(
            inteded_clusters=list(cluster_assignments.keys()),
            table_clusters=list(np.unique(new_table.new_clusters)))

        # check no cells were dropped
        assert len(cell_table[settings.CELL_LABEL]) == len(new_table[settings.CELL_LABEL])

        # check error raise when new_cluster_col already exists
        with pytest.raises(ValueError):
            post_cluster_utils.generate_new_cluster_resolution(
                cell_table, cluster_col=settings.CELL_TYPE, new_cluster_col="new_clusters",
                cluster_mapping=cluster_assignments, save_path=new_path)

        # check error raise when cell types missing from assignment dict
        with pytest.raises(ValueError):
            missing_assignments = {'A': ['A'], 'C': ['C']}
            post_cluster_utils.generate_new_cluster_resolution(
                cell_table, cluster_col=settings.CELL_TYPE, new_cluster_col="new_clusters_bad",
                cluster_mapping=missing_assignments, save_path=new_path)
