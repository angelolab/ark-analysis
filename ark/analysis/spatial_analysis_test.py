import tempfile

import numpy as np
import pandas as pd
import pytest

import ark.settings as settings
from ark.analysis import spatial_analysis
from ark.utils import load_utils, spatial_analysis_utils, test_utils

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

    with tempfile.TemporaryDirectory() as label_dir:

        test_utils._write_labels(label_dir, ["fov8", "fov9"], ["segmentation_label"], (10, 10),
                                 '', True, np.uint8, suffix='_feature_0')

        label_maps = load_utils.load_imgs_from_dir(label_dir, trim_suffix="_feature_0",
                                                   xr_channel_names=["segmentation_label"])

        dist_mats = spatial_analysis_utils.calc_dist_matrix(label_maps)

        all_data = test_utils.spoof_cell_table_from_labels(label_maps)

        vals_pos, stats_pos = \
            spatial_analysis.generate_channel_spatial_enrichment_stats(
                label_dir, marker_thresholds, all_data, excluded_channels=EXCLUDE_CHANNELS,
                bootstrap_num=100, dist_lim=100
            )

        # both fov8 and fov9 should be returned
        assert len(vals_pos) == 2

        vals_pos_fov8, stats_pos_fov8 = \
            spatial_analysis.generate_channel_spatial_enrichment_stats(
                label_dir, marker_thresholds, all_data, excluded_channels=EXCLUDE_CHANNELS,
                bootstrap_num=100, dist_lim=100, included_fovs=["fov8"]
            )

        # the fov8 values in vals_pos_fov8 should be the same as in vals_pos
        np.testing.assert_equal(vals_pos_fov8[0][0], vals_pos[0][0])

        # only fov8 should be returned
        assert len(vals_pos_fov8) == 1


def test_generate_cluster_spatial_enrichment_stats():

    # since the functionality if channel spatial enrichment is tested later,
    # only the number of elements returned and the included_fovs argument needs testing
    with tempfile.TemporaryDirectory() as label_dir:
        test_utils._write_labels(label_dir, ["fov8", "fov9"], ["segmentation_label"], (10, 10),
                                 '', True, np.uint8, suffix='_feature_0')

        label_maps = load_utils.load_imgs_from_dir(label_dir, trim_suffix="_feature_0",
                                                   xr_channel_names=["segmentation_label"])

        dist_mats = spatial_analysis_utils.calc_dist_matrix(label_maps)

        all_data = test_utils.spoof_cell_table_from_labels(label_maps)

        vals_pos, stats_pos = \
            spatial_analysis.generate_cluster_spatial_enrichment_stats(
                label_dir, all_data, bootstrap_num=100, dist_lim=100
            )

        # both fov8 and fov9 should be returned
        assert len(vals_pos) == 2

        vals_pos_fov8, stats_pos_fov8 = \
            spatial_analysis.generate_cluster_spatial_enrichment_stats(
                label_dir, all_data, bootstrap_num=100, dist_lim=100, included_fovs=["fov8"]
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

    _, stats_pos = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_pos, marker_thresholds, all_data_pos,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
            dist_lim=dist_lim)

    # Test both fov8 and fov9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for positive
    # enrichment as tested against a random set of distances between centroids
    assert stats_pos.loc["fov8", "p_pos", 2, 3] < .05
    assert stats_pos.loc["fov8", "p_neg", 2, 3] > .05
    assert stats_pos.loc["fov8", "z", 2, 3] > 0

    assert stats_pos.loc["fov9", "p_pos", 3, 2] < .05
    assert stats_pos.loc["fov9", "p_neg", 3, 2] > .05
    assert stats_pos.loc["fov9", "z", 3, 2] > 0

    # Negative enrichment
    all_data_neg, dist_mat_neg = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="negative", dist_lim=dist_lim)

    _, stats_neg = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_neg, marker_thresholds, all_data_neg,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
            dist_lim=dist_lim)

    # Test both fov8 and fov9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for negative
    # enrichment as tested against a random set of distances between centroids

    assert stats_neg.loc["fov8", "p_neg", 2, 3] < .05
    assert stats_neg.loc["fov8", "p_pos", 2, 3] > .05
    assert stats_neg.loc["fov8", "z", 2, 3] < 0

    assert stats_neg.loc["fov9", "p_neg", 3, 2] < .05
    assert stats_neg.loc["fov9", "p_pos", 3, 2] > .05
    assert stats_neg.loc["fov9", "z", 3, 2] < 0

    # No enrichment
    all_data_no_enrich, dist_mat_no_enrich = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="none", dist_lim=dist_lim)

    _, stats_no_enrich = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_no_enrich, marker_thresholds, all_data_no_enrich,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
            dist_lim=dist_lim)

    # Test both fov8 and fov9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for no enrichment
    # as tested against a random set of distances between centroids
    assert stats_no_enrich.loc["fov8", "p_pos", 2, 3] > .05
    assert stats_no_enrich.loc["fov8", "p_neg", 2, 3] > .05
    assert abs(stats_no_enrich.loc["fov8", "z", 2, 3]) < 2

    assert stats_no_enrich.loc["fov9", "p_pos", 3, 2] > .05
    assert stats_no_enrich.loc["fov9", "p_neg", 3, 2] > .05
    assert abs(stats_no_enrich.loc["fov9", "z", 3, 2]) < 2

    # run basic coverage check on context dependence code
    all_data_context, dist_mat_context = \
        test_utils._make_context_dist_exp_mats_spatial_test(dist_lim)

    _, _ = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_context, marker_thresholds, all_data_context,
            excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100, dist_lim=dist_lim,
            context_col='context_col'
        )

    # error checking
    with pytest.raises(ValueError):
        # attempt to exclude a column name that doesn't appear in the expression matrix
        _, stats_no_enrich = \
            spatial_analysis.calculate_channel_spatial_enrichment(
                dist_mat_no_enrich, marker_thresholds, all_data_no_enrich,
                excluded_channels=["bad_excluded_chan_name"], bootstrap_num=100,
                dist_lim=dist_lim)

    with pytest.raises(ValueError):
        # attempt to include fovs that do not exist
        _, stat_no_enrich = \
            spatial_analysis.calculate_channel_spatial_enrichment(
                dist_mat_no_enrich, marker_thresholds, all_data_no_enrich,
                excluded_channels=EXCLUDE_CHANNELS, included_fovs=[1, 100000],
                bootstrap_num=100, dist_lim=dist_lim)

    with pytest.raises(ValueError):
        # attempt to include marker thresholds and marker columns that do not exist
        bad_marker_thresholds = pd.DataFrame(np.zeros((21, 2)))
        bad_marker_thresholds.iloc[:, 1] = .5
        bad_marker_thresholds.iloc[:, 0] = np.arange(10, 31) + 2

        _, stat_no_enrich = \
            spatial_analysis.calculate_channel_spatial_enrichment(
                dist_mat_no_enrich, bad_marker_thresholds, all_data_no_enrich,
                excluded_channels=EXCLUDE_CHANNELS, bootstrap_num=100,
                dist_lim=dist_lim)


def test_calculate_cluster_spatial_enrichment():
    # Test z and p values
    dist_lim = 100

    # Positive enrichment
    all_data_pos, dist_mat_pos = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="positive", dist_lim=dist_lim)

    _, stats_pos = \
        spatial_analysis.calculate_cluster_spatial_enrichment(
            all_data_pos, dist_mat_pos,
            bootstrap_num=dist_lim, dist_lim=dist_lim)

    # Test both fov8 and fov9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for positive
    # enrichment as tested against a random set of distances between centroids
    assert stats_pos.loc["fov8", "p_pos", "Pheno1", "Pheno2"] < .05
    assert stats_pos.loc["fov8", "p_neg", "Pheno1", "Pheno2"] > .05
    assert stats_pos.loc["fov8", "z", "Pheno1", "Pheno2"] > 0

    assert stats_pos.loc["fov9", "p_pos", "Pheno2", "Pheno1"] < .05
    assert stats_pos.loc["fov9", "p_neg", "Pheno2", "Pheno1"] > .05
    assert stats_pos.loc["fov9", "z", "Pheno2", "Pheno1"] > 0

    # Negative enrichment
    all_data_neg, dist_mat_neg = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="negative", dist_lim=dist_lim)
    _, stats_neg = \
        spatial_analysis.calculate_cluster_spatial_enrichment(
            all_data_neg, dist_mat_neg,
            bootstrap_num=dist_lim, dist_lim=dist_lim)

    # Test both fov8 and fov9
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for negative
    # enrichment as tested against a random set of distances between centroids
    assert stats_neg.loc["fov8", "p_neg", "Pheno1", "Pheno2"] < .05
    assert stats_neg.loc["fov8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert stats_neg.loc["fov8", "z", "Pheno1", "Pheno2"] < 0

    assert stats_neg.loc["fov9", "p_neg", "Pheno2", "Pheno1"] < .05
    assert stats_neg.loc["fov9", "p_pos", "Pheno2", "Pheno1"] > .05
    assert stats_neg.loc["fov9", "z", "Pheno2", "Pheno1"] < 0

    all_data_no_enrich, dist_mat_no_enrich = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="none", dist_lim=dist_lim)

    _, stats_no_enrich = \
        spatial_analysis.calculate_cluster_spatial_enrichment(
            all_data_no_enrich, dist_mat_no_enrich,
            bootstrap_num=dist_lim, dist_lim=dist_lim)
    # Extract the p-values and z-scores of the distance of marker 1 vs marker 2 for no enrichment
    # as tested against a random set of distances between centroids
    assert stats_no_enrich.loc["fov8", "p_pos", "Pheno1", "Pheno2"] > .05
    assert stats_no_enrich.loc["fov8", "p_neg", "Pheno1", "Pheno2"] > .05
    assert abs(stats_no_enrich.loc["fov8", "z", "Pheno1", "Pheno2"]) < 2

    assert stats_no_enrich.loc["fov9", "p_pos", "Pheno2", "Pheno1"] > .05
    assert stats_no_enrich.loc["fov9", "p_neg", "Pheno2", "Pheno1"] > .05
    assert abs(stats_no_enrich.loc["fov9", "z", "Pheno2", "Pheno1"]) < 2

    # run basic coverage check on context dependence code
    all_data_context, dist_mat_context = \
        test_utils._make_context_dist_exp_mats_spatial_test(dist_lim)

    _, _ = \
        spatial_analysis.calculate_cluster_spatial_enrichment(
            all_data_context, dist_mat_context,
            bootstrap_num=dist_lim, dist_lim=dist_lim, context_col='context_col'
        )

    # run basic coverage check on feature distance
    all_data_hack, dist_mat_hack = \
        test_utils._make_dist_exp_mats_dist_feature_spatial_test(dist_lim)

    _, _ = \
        spatial_analysis.calculate_cluster_spatial_enrichment(
            all_data_hack, dist_mat_hack,
            bootstrap_num=dist_lim, dist_lim=dist_lim, distance_cols=['dist_feature_0']
        )

    # error checking
    with pytest.raises(ValueError):
        # attempt to include fovs that do not exist
        _, stats_no_enrich = \
            spatial_analysis.calculate_cluster_spatial_enrichment(
                all_data_no_enrich, dist_mat_no_enrich, included_fovs=[1, 100000],
                bootstrap_num=100, dist_lim=dist_lim)


def test_create_neighborhood_matrix():
    # get positive expression and distance matrices
    all_data_pos, dist_mat_pos = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="positive", dist_lim=51)

    # test if self_neighbor is False (default)
    counts, freqs = spatial_analysis.create_neighborhood_matrix(
        all_data_pos, dist_mat_pos, distlim=51
    )

    # test the counts values
    assert (counts[(counts[settings.FOV_ID] == "fov8") &
                   (counts[settings.CELL_LABEL].isin(range(1, 9)))]["Pheno1"] == 0).all()
    assert (counts[(counts[settings.FOV_ID] == "fov8") &
                   (counts[settings.CELL_LABEL].isin(range(1, 11)))]["Pheno2"] == 8).all()
    assert (counts[(counts[settings.FOV_ID] == "fov8") &
                   (counts[settings.CELL_LABEL].isin(range(11, 21)))]["Pheno1"] == 8).all()
    assert (counts[(counts[settings.FOV_ID] == "fov9") &
                   (counts[settings.CELL_LABEL].isin(range(1, 11)))]["Pheno3"] == 2).all()
    assert (counts[(counts[settings.FOV_ID] == "fov9") &
                   (counts[settings.CELL_LABEL].isin(range(11, 21)))]["Pheno1"] == 0).all()
    # test that cells with only itself as neighbor were removed from the table
    assert (len(counts[(counts[settings.FOV_ID] == "fov8") &
                       (counts[settings.CELL_LABEL].isin(range(21, 80)))]) == 0)

    # test if self_neighbor is True
    counts, freqs = spatial_analysis.create_neighborhood_matrix(
        all_data_pos, dist_mat_pos, distlim=51, self_neighbor=True
    )

    # test the counts values
    assert (counts[(counts[settings.FOV_ID] == "fov8") &
                   (counts[settings.CELL_LABEL].isin(range(1, 9)))]["Pheno1"] == 1).all()
    assert (counts[(counts[settings.FOV_ID] == "fov9") &
                   (counts[settings.CELL_LABEL].isin(range(1, 9)))]["Pheno3"] == 2).all()
    assert (counts[(counts[settings.FOV_ID] == "fov9") &
                   (counts[settings.CELL_LABEL].isin(range(11, 19)))]["Pheno1"] == 1).all()

    # error checking
    with pytest.raises(ValueError):
        # attempt to include fovs that do not exist
        counts, freqs = spatial_analysis.create_neighborhood_matrix(
            all_data_pos, dist_mat_pos, included_fovs=[1, 100000], distlim=51
        )


def test_generate_cluster_matrix_results():
    excluded_channels = ["Background", "HH3", "summed_channel"]

    all_data_pos, dist_mat_pos = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="positive", dist_lim=50
    )

    # we need corresponding dimensions, so use this method to generate
    # the neighborhood matrix
    neighbor_counts, neighbor_freqs = spatial_analysis.create_neighborhood_matrix(
        all_data_pos, dist_mat_pos, distlim=51
    )

    # error checking
    with pytest.raises(ValueError):
        # pass bad columns
        spatial_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=2, excluded_channels=["bad_col"]
        )

    with pytest.raises(ValueError):
        # include bad fovs
        spatial_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=2, excluded_channels=excluded_channels,
            included_fovs=[1000]
        )

    with pytest.raises(ValueError):
        # specify bad k for clustering
        spatial_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=1, excluded_channels=excluded_channels
        )

    all_data_markers_clusters, num_cell_type_per_cluster, mean_marker_exp_per_cluster = \
        spatial_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=2, excluded_channels=excluded_channels
        )

    # make sure we created a cluster_labels column
    assert settings.KMEANS_CLUSTER in all_data_markers_clusters.columns.values

    # can't really assert specific locations of values because cluster assignment stochastic
    # check just indexes and shapes
    assert num_cell_type_per_cluster.shape == (2, 3)
    assert list(num_cell_type_per_cluster.index.values) == ["Cluster1", "Cluster2"]
    assert list(num_cell_type_per_cluster.columns.values) == ["Pheno1", "Pheno2", "Pheno3"]

    assert mean_marker_exp_per_cluster.shape == (2, 20)
    assert list(mean_marker_exp_per_cluster.index.values) == ["Cluster1", "Cluster2"]
    assert list(mean_marker_exp_per_cluster.columns.values) == \
        list(np.arange(2, 14)) + list(np.arange(15, 23))

    # test excluded_channels=None
    all_data_markers_clusters, num_cell_type_per_cluster, mean_marker_exp_per_cluster = \
        spatial_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=2, excluded_channels=None
        )
    assert all(x in mean_marker_exp_per_cluster.columns.values for x in excluded_channels)


def test_compute_cluster_metrics_inertia():
    # get an example neighborhood matrix
    neighbor_mat = test_utils._make_neighborhood_matrix()

    # error checking
    with pytest.raises(ValueError):
        # pass an invalid k
        spatial_analysis.compute_cluster_metrics_inertia(neighbor_mat=neighbor_mat, min_k=1)

    with pytest.raises(ValueError):
        # pass an invalid k
        spatial_analysis.compute_cluster_metrics_inertia(neighbor_mat=neighbor_mat, max_k=1)

    with pytest.raises(ValueError):
        # pass invalid fovs
        spatial_analysis.compute_cluster_metrics_inertia(neighbor_mat=neighbor_mat,
                                                         included_fovs=["fov3"])

    # explicitly include fovs
    neighbor_cluster_stats = spatial_analysis.compute_cluster_metrics_inertia(
        neighbor_mat=neighbor_mat, max_k=3, included_fovs=["fov1", "fov2"])

    neighbor_cluster_stats = spatial_analysis.compute_cluster_metrics_inertia(
        neighbor_mat=neighbor_mat, max_k=3)

    # assert dimensions are correct
    assert len(neighbor_cluster_stats.values) == 2
    assert list(neighbor_cluster_stats.coords["cluster_num"]) == [2, 3]

    # assert k=3 produces the best inertia score
    last_k = neighbor_cluster_stats.loc[3].values
    assert np.all(last_k <= neighbor_cluster_stats.values)


def test_compute_cluster_metrics_silhouette():
    # get an example neighborhood matrix
    neighbor_mat = test_utils._make_neighborhood_matrix()

    # error checking
    with pytest.raises(ValueError):
        # pass an invalid k
        spatial_analysis.compute_cluster_metrics_silhouette(neighbor_mat=neighbor_mat, min_k=1)

    with pytest.raises(ValueError):
        # pass an invalid k
        spatial_analysis.compute_cluster_metrics_silhouette(neighbor_mat=neighbor_mat, max_k=1)

    with pytest.raises(ValueError):
        # pass invalid fovs
        spatial_analysis.compute_cluster_metrics_silhouette(neighbor_mat=neighbor_mat,
                                                            included_fovs=["fov3"])

    # explicitly include fovs
    neighbor_cluster_stats = spatial_analysis.compute_cluster_metrics_silhouette(
        neighbor_mat=neighbor_mat, max_k=3, included_fovs=["fov1", "fov2"])

    # test subsampling
    neighbor_cluster_stats = spatial_analysis.compute_cluster_metrics_silhouette(
        neighbor_mat=neighbor_mat, max_k=3, subsample=10)

    neighbor_cluster_stats = spatial_analysis.compute_cluster_metrics_silhouette(
        neighbor_mat=neighbor_mat, max_k=3)

    # assert dimensions are correct
    assert len(neighbor_cluster_stats.values) == 2
    assert list(neighbor_cluster_stats.coords["cluster_num"]) == [2, 3]

    # assert k=3 produces the best silhouette score for both fov1 and fov2
    last_k = neighbor_cluster_stats.loc[3].values
    assert np.all(last_k >= neighbor_cluster_stats.values)
