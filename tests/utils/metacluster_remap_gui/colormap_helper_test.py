import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import ListedColormap
from alpineer import misc_utils

from ark.utils.metacluster_remap_gui.colormap_helper import (distinct_cmap, distinct_rgbs,
                                                             generate_meta_cluster_colormap_dict)


def test_colormap_is_distinct():
    assert len(set(distinct_rgbs(200))) == 200


def test_colormap_runs():
    distinct_cmap(10)


def test_generate_meta_cluster_colormap_dict():
    with tempfile.TemporaryDirectory() as temp_dir:
        # basic error check: bad remapping path
        with pytest.raises(FileNotFoundError):
            generate_meta_cluster_colormap_dict(
                os.path.join(temp_dir, 'bad_remap_path.csv'), None
            )

        # basic error check: remapping data contains bad columns
        with pytest.raises(ValueError):
            bad_sample_remapping = {
                'cluster': [i for i in np.arange(10)],
                'metacluster': [int(i / 50) for i in np.arange(100)],
                'mc_name_bad': ['meta' + str(int(i / 50)) for i in np.arange(100)]
            }

            bad_sample_remapping = pd.DataFrame.from_dict(bad_sample_remapping)
            bad_sample_remapping.to_csv(
                os.path.join(temp_dir, 'bad_sample_remapping.csv'),
                index=False
            )

            generate_meta_cluster_colormap_dict(
                os.path.join(temp_dir, 'bad_sample_remapping.csv'), None
            )

        # define a dummy remapping
        sample_remapping = {
            'pixel_som_cluster': [i for i in np.arange(100)],
            'pixel_meta_cluster': [int(i / 50) + 1 for i in np.arange(100)],
            'pixel_meta_cluster_rename': ['meta' + str(int(i / 50) + 1) for i in np.arange(100)]
        }

        sample_remapping = pd.DataFrame.from_dict(sample_remapping)
        sample_remapping.to_csv(
            os.path.join(temp_dir, 'sample_remapping.csv'),
            index=False
        )

        # define a sample ListedColormap
        cmap = ListedColormap(['red', 'blue', 'green'])

        raw_cmap, renamed_cmap = generate_meta_cluster_colormap_dict(
            os.path.join(temp_dir, 'sample_remapping.csv'), cmap
        )

        # assert the correct meta cluster labels are contained in both dicts
        misc_utils.verify_same_elements(
            raw_cmap_keys=list(raw_cmap.keys()),
            raw_meta_clusters=sample_remapping['pixel_meta_cluster'].values
        )
        misc_utils.verify_same_elements(
            renamed_cmap_keys=list(renamed_cmap.keys()),
            renamed_meta_clusters=sample_remapping['pixel_meta_cluster_rename'].values
        )

        # assert the colors match up
        assert raw_cmap[1] == renamed_cmap['meta1'] == (1.0, 0.0, 0.0, 1.0)
        assert raw_cmap[2] == renamed_cmap['meta2'] == (0.0, 0.0, 1.0, 1.0)
