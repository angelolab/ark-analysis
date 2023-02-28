import tempfile

import numpy as np
import pandas as pd
import pytest
from alpineer import load_utils
from alpineer.test_utils import _write_labels

import ark.settings as settings
from ark.analysis import spatial_enrichment, spatial_analysis_utils
import test_utils


EXCLUDE_CHANNELS = [
    "Background",
    "HH3",
    "summed_channel",
]

DEFAULT_COLUMNS = \
    [settings.CELL_SIZE] \
    + list(range(1, 24)) \
    + [
        settings.CELL_LABEL,
        'area',
        'eccentricity',
        'maj_axis_length',
        'min_axis_length',
        'perimiter',
        settings.FOV_ID,
        settings.CELL_TYPE,
    ]
list(map(
    DEFAULT_COLUMNS.__setitem__, [1, 14, 23], EXCLUDE_CHANNELS
))


def test_generate_channel_spatial_enrichment_stats():
    # since the functionality of channel spatial enrichment is tested later,
    # only the number of elements returned and the included_fovs argument needs testing
    marker_thresholds = test_utils._make_threshold_mat(in_utils=False)

    with tempfile.TemporaryDirectory() as label_dir, \
         tempfile.TemporaryDirectory() as dist_mat_dir:
        _write_labels(label_dir, ["fov8", "fov9"], ["segmentation_label"], (10, 10),
                      '', True, np.uint8, suffix='_whole_cell')

        spatial_analysis_utils.calc_dist_matrix(label_dir, dist_mat_dir)
        label_maps = load_utils.load_imgs_from_dir(label_dir, trim_suffix="_whole_cell",
                                                   xr_channel_names=["segmentation_label"])
        all_data = test_utils.spoof_cell_table_from_labels(label_maps)

        vals_pos, stats_pos = \
            spatial_enrichment.generate_channel_spatial_enrichment_stats(
                label_dir, dist_mat_dir, marker_thresholds, all_data,
                excluded_channels=EXCLUDE_CHANNELS,
                bootstrap_num=100, dist_lim=100
            )

        # both fov8 and fov9 should be returned
        assert len(vals_pos) == 2

        vals_pos_fov8, stats_pos_fov8 = \
            spatial_enrichment.generate_channel_spatial_enrichment_stats(
                label_dir, dist_mat_dir, marker_thresholds, all_data,
                excluded_channels=EXCLUDE_CHANNELS,
                bootstrap_num=100, dist_lim=100, included_fovs=["fov8"]
            )

        # the fov8 values in vals_pos_fov8 should be the same as in vals_pos
        np.testing.assert_equal(vals_pos_fov8[0][0], vals_pos[0][0])

        # only fov8 should be returned
        assert len(vals_pos_fov8) == 1


def test_generate_cluster_spatial_enrichment_stats():
    # since the functionality if channel spatial enrichment is tested later,
    # only the number of elements returned and the included_fovs argument needs testing
    with tempfile.TemporaryDirectory() as label_dir, \
         tempfile.TemporaryDirectory() as dist_mat_dir:
        _write_labels(label_dir, ["fov8", "fov9"], ["segmentation_label"], (10, 10),
                      '', True, np.uint8, suffix='_whole_cell')

        spatial_analysis_utils.calc_dist_matrix(label_dir, dist_mat_dir)
        label_maps = load_utils.load_imgs_from_dir(label_dir, trim_suffix="_whole_cell",
                                                   xr_channel_names=["segmentation_label"])
        all_data = test_utils.spoof_cell_table_from_labels(label_maps)

        vals_pos, stats_pos = \
            spatial_enrichment.generate_cluster_spatial_enrichment_stats(
                label_dir, dist_mat_dir, all_data,
                bootstrap_num=100, dist_lim=100
            )

        # both fov8 and fov9 should be returned
        assert len(vals_pos) == 2

        vals_pos_fov8, stats_pos_fov8 = \
            spatial_enrichment.generate_cluster_spatial_enrichment_stats(
                label_dir, dist_mat_dir, all_data,
                bootstrap_num=100, dist_lim=100, included_fovs=["fov8"]
            )

        # the fov8 values in vals_pos_fov8 should be the same as in vals_pos
        np.testing.assert_equal(vals_pos_fov8[0][0], vals_pos[0][0])

        # only fov8 should be returned
        assert len(vals_pos_fov8) == 1


def test_calculate_channel_spatial_enrichment():
    dist_lim = 100

    # Test z and p values
    marker_thresholds = test_utils._make_threshold_mat(in_utils=False)

    # Positive enrichment
    all_data_pos, dist_mat_pos = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="positive", dist_lim=dist_lim)

    _, stats_pos8 = \
        spatial_enrichment.calculate_channel_spatial_enrichment(
            'fov8', dist_mat_pos['fov8'], marker_thresholds, all_data_pos,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
            dist_lim=dist_lim)

    _, stats_pos9 = \
        spatial_enrichment.calculate_channel_spatial_enrichment(
            'fov9', dist_mat_pos['fov9'], marker_thresholds, all_data_pos,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
            dist_lim=dist_lim)

    # Test both fov8 and fov9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for positive
    # enrichment as tested against a random set of distances between centroids
    assert stats_pos8.loc["fov8", "p_pos", 2, 3] < .05
    assert stats_pos8.loc["fov8", "p_neg", 2, 3] > .05
    assert stats_pos8.loc["fov8", "z", 2, 3] > 0

    assert stats_pos9.loc["fov9", "p_pos", 3, 2] < .05
    assert stats_pos9.loc["fov9", "p_neg", 3, 2] > .05
    assert stats_pos9.loc["fov9", "z", 3, 2] > 0

    # Negative enrichment
    all_data_neg, dist_mat_neg = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="negative", dist_lim=dist_lim)

    _, stats_neg8 = \
        spatial_enrichment.calculate_channel_spatial_enrichment(
            'fov8', dist_mat_neg['fov8'], marker_thresholds, all_data_neg,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
            dist_lim=dist_lim)

    _, stats_neg9 = \
        spatial_enrichment.calculate_channel_spatial_enrichment(
            'fov9', dist_mat_neg['fov9'], marker_thresholds, all_data_neg,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
            dist_lim=dist_lim)

    # Test both fov8 and fov9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for negative
    # enrichment as tested against a random set of distances between centroids

    assert stats_neg8.loc["fov8", "p_neg", 2, 3] < .05
    assert stats_neg8.loc["fov8", "p_pos", 2, 3] > .05
    assert stats_neg8.loc["fov8", "z", 2, 3] < 0

    assert stats_neg9.loc["fov9", "p_neg", 3, 2] < .05
    assert stats_neg9.loc["fov9", "p_pos", 3, 2] > .05
    assert stats_neg9.loc["fov9", "z", 3, 2] < 0

    # No enrichment
    all_data_no_enrich, dist_mat_no_enrich = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="none", dist_lim=dist_lim)

    _, stats_no_enrich8 = \
        spatial_enrichment.calculate_channel_spatial_enrichment(
            'fov8', dist_mat_no_enrich['fov8'], marker_thresholds, all_data_no_enrich,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
            dist_lim=dist_lim)

    _, stats_no_enrich9 = \
        spatial_enrichment.calculate_channel_spatial_enrichment(
            'fov9', dist_mat_no_enrich['fov9'], marker_thresholds, all_data_no_enrich,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
            dist_lim=dist_lim)

    # Test both fov8 and fov9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for no enrichment
    # as tested against a random set of distances between centroids
    assert stats_no_enrich8.loc["fov8", "p_pos", 2, 3] > .05
    assert stats_no_enrich8.loc["fov8", "p_neg", 2, 3] > .05
    assert abs(stats_no_enrich8.loc["fov8", "z", 2, 3]) < 2

    assert stats_no_enrich9.loc["fov9", "p_pos", 3, 2] > .05
    assert stats_no_enrich9.loc["fov9", "p_neg", 3, 2] > .05
    assert abs(stats_no_enrich9.loc["fov9", "z", 3, 2]) < 2

    # run basic coverage check on context dependence code
    all_data_context, dist_mat_context = \
        test_utils._make_context_dist_exp_mats_spatial_test(dist_lim)

    _, _ = \
        spatial_enrichment.calculate_channel_spatial_enrichment(
            'fov8', dist_mat_context['fov8'], marker_thresholds, all_data_context,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100, dist_lim=dist_lim,
            context_col='context_col'
        )

    _, _ = \
        spatial_enrichment.calculate_channel_spatial_enrichment(
            'fov9', dist_mat_context['fov9'], marker_thresholds, all_data_context,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100, dist_lim=dist_lim,
            context_col='context_col'
        )

    # error checking
    with pytest.raises(ValueError):
        # attempt to exclude a column name that doesn't appear in the expression matrix
        _, stats_no_enrich = \
            spatial_enrichment.calculate_channel_spatial_enrichment(
                'fov8', dist_mat_no_enrich['fov8'], marker_thresholds, all_data_no_enrich,
                excluded_channels=["bad_excluded_chan_name"], bootstrap_num=100,
                dist_lim=dist_lim)

    with pytest.raises(ValueError):
        # attempt to include a fov that doesn't exist
        _, stat_no_enrich = \
            spatial_enrichment.calculate_channel_spatial_enrichment(
                'fov10', dist_mat_no_enrich['fov8'], marker_thresholds, all_data_no_enrich,
                excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
                dist_lim=dist_lim)

    with pytest.raises(ValueError):
        # attempt to include marker thresholds and marker columns that do not exist
        bad_marker_thresholds = pd.DataFrame(np.zeros((21, 2)), columns=["marker", "threshold"])
        bad_marker_thresholds.iloc[:, 1] = .5
        bad_marker_thresholds.iloc[:, 0] = np.arange(10, 31) + 2

        _, stat_no_enrich = \
            spatial_enrichment.calculate_channel_spatial_enrichment(
                'fov8', dist_mat_no_enrich['fov8'], bad_marker_thresholds, all_data_no_enrich,
                excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
                dist_lim=dist_lim)


def test_calculate_cluster_spatial_enrichment():
    # Test z and p values
    dist_lim = 100

    # Positive enrichment
    all_data_pos, dist_mat_pos = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="positive", dist_lim=dist_lim)

    _, stats_pos8 = \
        spatial_enrichment.calculate_cluster_spatial_enrichment(
            'fov8', all_data_pos, dist_mat_pos['fov8'],
            bootstrap_num=dist_lim, dist_lim=dist_lim)

    _, stats_pos9 = \
        spatial_enrichment.calculate_cluster_spatial_enrichment(
            'fov9', all_data_pos, dist_mat_pos['fov9'],
            bootstrap_num=dist_lim, dist_lim=dist_lim)

    # Test both fov8 and fov9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for positive
    # enrichment as tested against a random set of distances between centroids
    assert stats_pos8.loc["fov8", "p_pos", "Pheno1", "Pheno2"] < .05
    assert stats_pos8.loc["fov8", "p_neg", "Pheno1", "Pheno2"] > .05
    assert stats_pos8.loc["fov8", "z", "Pheno1", "Pheno2"] > 0

    assert stats_pos9.loc["fov9", "p_pos", "Pheno2", "Pheno1"] < .05
    assert stats_pos9.loc["fov9", "p_neg", "Pheno2", "Pheno1"] > .05
    assert stats_pos9.loc["fov9", "z", "Pheno2", "Pheno1"] > 0

    # Negative enrichment
    all_data_neg, dist_mat_neg = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="negative", dist_lim=dist_lim)

    _, stats_neg8 = \
        spatial_enrichment.calculate_cluster_spatial_enrichment(
            'fov8', all_data_neg, dist_mat_neg['fov8'],
            bootstrap_num=dist_lim, dist_lim=dist_lim)

    _, stats_neg9 = \
        spatial_enrichment.calculate_cluster_spatial_enrichment(
            'fov9', all_data_neg, dist_mat_neg['fov9'],
            bootstrap_num=dist_lim, dist_lim=dist_lim)

    # Test both fov8 and fov9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for negative
    # enrichment as tested against a random set of distances between centroids
    assert stats_neg8.loc["fov8", "p_neg", "Pheno1", "Pheno2"] < .05
    assert stats_neg8.loc["fov8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert stats_neg8.loc["fov8", "z", "Pheno1", "Pheno2"] < 0

    assert stats_neg9.loc["fov9", "p_neg", "Pheno2", "Pheno1"] < .05
    assert stats_neg9.loc["fov9", "p_pos", "Pheno2", "Pheno1"] > .05
    assert stats_neg9.loc["fov9", "z", "Pheno2", "Pheno1"] < 0

    all_data_no_enrich, dist_mat_no_enrich = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="none", dist_lim=dist_lim)

    _, stats_no_enrich8 = \
        spatial_enrichment.calculate_cluster_spatial_enrichment(
            'fov8', all_data_no_enrich, dist_mat_no_enrich['fov8'],
            bootstrap_num=dist_lim, dist_lim=dist_lim)

    _, stats_no_enrich9 = \
        spatial_enrichment.calculate_cluster_spatial_enrichment(
            'fov9', all_data_no_enrich, dist_mat_no_enrich['fov9'],
            bootstrap_num=dist_lim, dist_lim=dist_lim)

    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for no enrichment
    # as tested against a random set of distances between centroids
    assert stats_no_enrich8.loc["fov8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert stats_no_enrich8.loc["fov8", "p_neg", "Pheno1", "Pheno2"] > .05
    assert abs(stats_no_enrich8.loc["fov8", "z", "Pheno1", "Pheno2"]) < 2

    assert stats_no_enrich9.loc["fov9", "p_pos", "Pheno2", "Pheno1"] > .05
    assert stats_no_enrich9.loc["fov9", "p_neg", "Pheno2", "Pheno1"] > .05
    assert abs(stats_no_enrich9.loc["fov9", "z", "Pheno2", "Pheno1"]) < 2

    # run basic coverage check on context dependence code
    all_data_context, dist_mat_context = \
        test_utils._make_context_dist_exp_mats_spatial_test(dist_lim)

    _, _ = \
        spatial_enrichment.calculate_cluster_spatial_enrichment(
            'fov8', all_data_context, dist_mat_context['fov8'],
            bootstrap_num=dist_lim, dist_lim=dist_lim, context_col='context_col'
        )

    _, _ = \
        spatial_enrichment.calculate_cluster_spatial_enrichment(
            'fov9', all_data_context, dist_mat_context['fov9'],
            bootstrap_num=dist_lim, dist_lim=dist_lim, context_col='context_col'
        )

    # run basic coverage check on feature distance
    all_data_hack, dist_mat_hack = \
        test_utils._make_dist_exp_mats_dist_feature_spatial_test(dist_lim)

    _, _ = \
        spatial_enrichment.calculate_cluster_spatial_enrichment(
            'fov8', all_data_hack, dist_mat_hack['fov8'],
            bootstrap_num=dist_lim, dist_lim=dist_lim, distance_cols=['dist_whole_cell']
        )

    _, _ = \
        spatial_enrichment.calculate_cluster_spatial_enrichment(
            'fov9', all_data_hack, dist_mat_hack['fov9'],
            bootstrap_num=dist_lim, dist_lim=dist_lim, distance_cols=['dist_whole_cell']
        )

    # error checking
    with pytest.raises(ValueError):
        # attempt to include a fov that doesn't exist
        _, stats_no_enrich = \
            spatial_enrichment.calculate_cluster_spatial_enrichment(
                'fov10', all_data_no_enrich, dist_mat_no_enrich['fov8'],
                bootstrap_num=100, dist_lim=dist_lim)
