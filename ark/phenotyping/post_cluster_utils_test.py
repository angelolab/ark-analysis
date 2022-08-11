import os
import pytest

import numpy as np
import pandas as pd

from ark.phenotyping import post_cluster_utils


def test_plot_hist_thresholds():
    pops = np.repeat(['pop1', 'pop2', 'pop3'], 5)
    marker_1 = np.random.rand(len(pops))

    cell_table = pd.DataFrame({'cell_meta_cluster': pops, 'marker_1': marker_1})

    # populations argument must be a list
    with pytest.raises(ValueError, match='must be a list'):
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


def test_create_updated_cell_masks(tmp_dir):
    seg_dir = os.path.join(tmp_dir, 'seg')
    fovs = ['fov1', 'fov2', 'fov3']

    for fov in fovs:
        data = np.random.randint(10, 1, 20).reshape(10, 10)