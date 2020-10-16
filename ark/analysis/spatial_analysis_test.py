import pytest
import numpy as np
import pandas as pd
import xarray as xr

from ark.analysis import spatial_analysis
from ark.utils import synthetic_spatial_datagen
from ark.utils import test_utils


def test_calculate_channel_spatial_enrichment():
    dist_lim = 100

    excluded_colnames = ["cell_size", "Background", "HH3",
                         "summed_channel", "cellLabelInImage", "area",
                         "eccentricity", "major_axis_length", "minor_axis_length",
                         "perimeter", "SampleID", "FlowSOM_ID", "cell_type"]

    # Test z and p values
    marker_thresholds = test_utils._make_threshold_mat(in_utils=False)

    # Positive enrichment
    all_data_pos, dist_mat_pos = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="positive", dist_lim=dist_lim)

    _, stats_pos = \
        spatial_analysis.calculate_channel_spatial_enrichment(
            dist_mat_pos, marker_thresholds, all_data_pos,
            excluded_colnames=excluded_colnames, bootstrap_num=100,
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
            excluded_colnames=excluded_colnames, bootstrap_num=100,
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
            excluded_colnames=excluded_colnames, bootstrap_num=100,
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

    # error checking
    with pytest.raises(ValueError):
        # attempt to exclude a column name that doesn't appear in the expression matrix
        _, stats_no_enrich = \
            spatial_analysis.calculate_channel_spatial_enrichment(
                dist_mat_no_enrich, marker_thresholds, all_data_no_enrich,
                excluded_colnames=["bad_excluded_col_name"], bootstrap_num=100,
                dist_lim=dist_lim)

    with pytest.raises(ValueError):
        # attempt to include fovs that do not exist
        _, stat_no_enrich = \
            spatial_analysis.calculate_channel_spatial_enrichment(
                dist_mat_no_enrich, marker_thresholds, all_data_no_enrich,
                excluded_colnames=excluded_colnames, included_fovs=[1, 100000],
                bootstrap_num=100, dist_lim=dist_lim)

    with pytest.raises(ValueError):
        # attempt to include marker thresholds and marker columns that do not exist
        bad_marker_thresholds = pd.DataFrame(np.zeros((21, 2)))
        bad_marker_thresholds.iloc[:, 1] = .5
        bad_marker_thresholds.iloc[:, 0] = np.arange(10, 31) + 2

        all_data_no_enrich_bad = all_data_no_enrich.rename(columns={18: 200})

        _, stat_no_enrich = \
            spatial_analysis.calculate_channel_spatial_enrichment(
                dist_mat_no_enrich, bad_marker_thresholds, all_data_no_enrich,
                excluded_colnames=excluded_colnames, bootstrap_num=100,
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

    counts, freqs = spatial_analysis.create_neighborhood_matrix(
        all_data_pos, dist_mat_pos, distlim=51
    )

    # Test the counts values for both fovs
    assert (counts.loc[:9, "Pheno2"] == 8).all()
    assert (counts.loc[10:19, "Pheno3"] == 8).all()

    assert (counts.loc[80:89, "Pheno3"] == 8).all()
    assert (counts.loc[90:99, "Pheno1"] == 8).all()

    # error checking
    with pytest.raises(ValueError):
        # attempt to include fovs that do not exist
        counts, freqs = spatial_analysis.create_neighborhood_matrix(
            all_data_pos, dist_mat_pos, included_fovs=[1, 100000], distlim=51
        )


def test_generate_cluster_matrix_results():
    excluded_colnames = ["cell_size", "Background", "HH3",
                         "summed_channel", "cellLabelInImage", "area",
                         "eccentricity", "major_axis_length", "minor_axis_length",
                         "perimeter", "SampleID", "FlowSOM_ID", "cell_type"]

    all_data_pos, dist_mat_pos = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="positive", dist_lim=50
    )

    # we need corresponding dimensions, so use this method to generate
    # the neighborhood matrix
    neighbor_counts, neighbor_freqs = spatial_analysis.create_neighborhood_matrix(
        all_data_pos, dist_mat_pos, distlim=51
    )

    neighbor_counts = neighbor_counts.drop("cellLabelInImage", axis=1)

    # error checking
    with pytest.raises(ValueError):
        # pass bad columns
        spatial_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=3, excluded_colnames=["bad_col"]
        )

    with pytest.raises(ValueError):
        # include bad fovs
        spatial_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=3, excluded_colnames=excluded_colnames,
            included_fovs=[1000]
        )

    with pytest.raises(ValueError):
        # specify bad k for clustering
        spatial_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=1, excluded_colnames=excluded_colnames
        )

    num_cell_type_per_cluster, mean_marker_exp_per_cluster = \
        spatial_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=3, excluded_colnames=excluded_colnames
        )

    # can't really assert specific locations of values because cluster assignment stochastic
    # check just indexes and shapes
    assert num_cell_type_per_cluster.shape == (3, 3)
    assert list(num_cell_type_per_cluster.index.values) == [0, 1, 2]
    assert list(num_cell_type_per_cluster.columns.values) == ["Pheno1", "Pheno2", "Pheno3"]

    assert mean_marker_exp_per_cluster.shape == (3, 20)
    assert list(mean_marker_exp_per_cluster.index.values) == [0, 1, 2]
    assert list(mean_marker_exp_per_cluster.columns.values) == \
        list(np.arange(2, 14)) + list(np.arange(15, 23))


def test_compute_cluster_metrics():
    # get an example neighborhood matrix
    neighbor_mat = test_utils._make_neighborhood_matrix()
    neighbor_mat = neighbor_mat.drop("cellLabelInImage", axis=1)

    # error checking
    with pytest.raises(ValueError):
        # pass an invalid k
        spatial_analysis.compute_cluster_metrics(neighbor_mat=neighbor_mat, max_k=1)

    with pytest.raises(ValueError):
        # pass invalid fovs
        spatial_analysis.compute_cluster_metrics(neighbor_mat=neighbor_mat,
                                                 included_fovs=["fov3"])

    neighbor_cluster_stats = spatial_analysis.compute_cluster_metrics(
        neighbor_mat=neighbor_mat, max_k=3)

    # assert dimensions are correct
    assert len(neighbor_cluster_stats.values) == 2
    assert list(neighbor_cluster_stats.coords["cluster_num"]) == [2, 3]

    # assert k=3 produces the best silhouette score for both fov1 and fov2
    last_k = neighbor_cluster_stats.loc[3].values
    assert np.all(last_k >= neighbor_cluster_stats.values)

    last_k = neighbor_cluster_stats.loc[3].values
    assert np.all(last_k >= neighbor_cluster_stats.values)
