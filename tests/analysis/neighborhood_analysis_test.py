import os
import tempfile
import math

import numpy as np
import pandas as pd
import pytest

import ark.settings as settings
from ark.analysis import neighborhood_analysis
import test_utils


def test_create_neighborhood_matrix():
    # get positive expression and distance matrices
    all_data_pos, dist_mat_pos = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="positive", dist_lim=51)

    with tempfile.TemporaryDirectory() as dist_mat_dir:
        for fov in dist_mat_pos:
            dist_mat_pos[fov].to_netcdf(
                os.path.join(dist_mat_dir, fov + '_dist_mat.xr'),
                format='NETCDF3_64BIT'
            )

        # error checking
        with pytest.raises(ValueError):
            # attempt to include fovs that do not exist
            counts, freqs = neighborhood_analysis.create_neighborhood_matrix(
                all_data_pos, dist_mat_dir, included_fovs=[1, 100000], distlim=51
            )

        # test if self_neighbor is False (default)
        counts, freqs = neighborhood_analysis.create_neighborhood_matrix(
            all_data_pos, dist_mat_dir, distlim=51
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

        # check that cell type is in matrix
        assert settings.CELL_TYPE in counts.columns

        # test if self_neighbor is True
        counts, freqs = neighborhood_analysis.create_neighborhood_matrix(
            all_data_pos, dist_mat_dir, distlim=51, self_neighbor=True
        )

        # test the counts values
        assert (counts[(counts[settings.FOV_ID] == "fov8") &
                       (counts[settings.CELL_LABEL].isin(range(1, 9)))]["Pheno1"] == 1).all()
        assert (counts[(counts[settings.FOV_ID] == "fov9") &
                       (counts[settings.CELL_LABEL].isin(range(1, 9)))]["Pheno3"] == 2).all()
        assert (counts[(counts[settings.FOV_ID] == "fov9") &
                       (counts[settings.CELL_LABEL].isin(range(11, 19)))]["Pheno1"] == 1).all()

        # test on a non-continuous index
        all_data_pos_sub = all_data_pos.iloc[np.r_[0:60, 80:140], :]
        dist_mat_pos_sub = {}
        dist_mat_pos_sub['fov8'] = dist_mat_pos['fov8'][0:60, 0:60]
        dist_mat_pos_sub['fov9'] = dist_mat_pos['fov9'][0:60, 0:60]

        for fov in dist_mat_pos:
            dist_mat_pos[fov].to_netcdf(
                os.path.join(dist_mat_dir, fov + '_dist_mat.xr'),
                format='NETCDF3_64BIT'
            )

        counts, freqs = neighborhood_analysis.create_neighborhood_matrix(
            all_data_pos, dist_mat_dir
        )

        # test the counts values
        assert (counts[(counts[settings.FOV_ID] == "fov8") &
                       (counts[settings.CELL_LABEL].isin(range(1, 9)))]["Pheno1"] == 0).all()
        assert (counts[(counts[settings.FOV_ID] == "fov9") &
                       (counts[settings.CELL_LABEL].isin(range(1, 9)))]["Pheno3"] == 2).all()
        assert (counts[(counts[settings.FOV_ID] == "fov9") &
                       (counts[settings.CELL_LABEL].isin(range(11, 19)))]["Pheno1"] == 0).all()


def test_generate_cluster_matrix_results():
    excluded_channels = ["Background", "HH3", "summed_channel"]

    all_data_pos, dist_mat_pos = test_utils._make_dist_exp_mats_spatial_test(
        enrichment_type="positive", dist_lim=50
    )

    # we need corresponding dimensions, so use this method to generate
    # the neighborhood matrix
    with tempfile.TemporaryDirectory() as dist_mat_dir:
        for fov in dist_mat_pos:
            dist_mat_pos[fov].to_netcdf(
                os.path.join(dist_mat_dir, fov + '_dist_mat.xr'),
                format='NETCDF3_64BIT'
            )

        neighbor_counts, neighbor_freqs = neighborhood_analysis.create_neighborhood_matrix(
            all_data_pos, dist_mat_dir, distlim=51
        )

    # error checking
    with pytest.raises(ValueError):
        # pass bad columns
        neighborhood_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=2, excluded_channels=["bad_col"]
        )

    with pytest.raises(ValueError):
        # include bad fovs
        neighborhood_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=2, excluded_channels=excluded_channels,
            included_fovs=[1000]
        )

    with pytest.raises(ValueError):
        # specify bad k for clustering
        neighborhood_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=1, excluded_channels=excluded_channels
        )

    all_data_markers_clusters, num_cell_type_per_cluster, mean_marker_exp_per_cluster = \
        neighborhood_analysis.generate_cluster_matrix_results(
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
        neighborhood_analysis.generate_cluster_matrix_results(
            all_data_pos, neighbor_counts, cluster_num=2, excluded_channels=None
        )
    assert all(x in mean_marker_exp_per_cluster.columns.values for x in excluded_channels)


def test_compute_cluster_metrics_inertia():
    # get an example neighborhood matrix
    neighbor_mat = test_utils._make_neighborhood_matrix()

    # error checking
    with pytest.raises(ValueError):
        # pass an invalid k
        neighborhood_analysis.compute_cluster_metrics_inertia(neighbor_mat=neighbor_mat, min_k=1)

    with pytest.raises(ValueError):
        # pass an invalid k
        neighborhood_analysis.compute_cluster_metrics_inertia(neighbor_mat=neighbor_mat, max_k=1)

    with pytest.raises(ValueError):
        # pass invalid fovs
        neighborhood_analysis.compute_cluster_metrics_inertia(neighbor_mat=neighbor_mat,
                                                              included_fovs=["fov3"])

    # explicitly include fovs
    neighbor_cluster_stats = neighborhood_analysis.compute_cluster_metrics_inertia(
        neighbor_mat=neighbor_mat, max_k=3, included_fovs=["fov1", "fov2"])

    neighbor_cluster_stats = neighborhood_analysis.compute_cluster_metrics_inertia(
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
        neighborhood_analysis.compute_cluster_metrics_silhouette(
            neighbor_mat=neighbor_mat, min_k=1)

    with pytest.raises(ValueError):
        # pass an invalid k
        neighborhood_analysis.compute_cluster_metrics_silhouette(
            neighbor_mat=neighbor_mat, max_k=1)

    with pytest.raises(ValueError):
        # pass invalid fovs
        neighborhood_analysis.compute_cluster_metrics_silhouette(neighbor_mat=neighbor_mat,
                                                                 included_fovs=["fov3"])

    # explicitly include fovs
    neighbor_cluster_stats = neighborhood_analysis.compute_cluster_metrics_silhouette(
        neighbor_mat=neighbor_mat, max_k=3, included_fovs=["fov1", "fov2"])

    # test subsampling
    neighbor_cluster_stats = neighborhood_analysis.compute_cluster_metrics_silhouette(
        neighbor_mat=neighbor_mat, max_k=3, subsample=10)

    neighbor_cluster_stats = neighborhood_analysis.compute_cluster_metrics_silhouette(
        neighbor_mat=neighbor_mat, max_k=3)

    # assert dimensions are correct
    assert len(neighbor_cluster_stats.values) == 2
    assert list(neighbor_cluster_stats.coords["cluster_num"]) == [2, 3]

    # assert k=3 produces the best silhouette score for both fov1 and fov2
    last_k = neighbor_cluster_stats.loc[3].values
    assert np.all(last_k >= neighbor_cluster_stats.values)


def test_compute_cell_ratios():
    cell_neighbors_mat = pd.DataFrame({
        settings.FOV_ID: ['fov1', 'fov1', 'fov1', 'fov1', 'fov1', 'fov1', 'fov1'],
        settings.CELL_LABEL: list(range(1, 8)),
        settings.CELL_TYPE: ['cell1', 'cell2', 'cell1', 'cell1', 'cell2', 'cell2', 'cell1'],
        'cell1': [1, 0, 2, 2, 1, 2, 0],
        'cell2': [1, 2, 1, 1, 2, 2, 0]
    })
    ratios = neighborhood_analysis.compute_cell_ratios(
        cell_neighbors_mat, ['cell1'], ['cell2'], ['fov1'])
    assert ratios.equals(pd.DataFrame({'fov': 'fov1', 'pop1_pop2_ratio': [4/3],
                                       'pop2_pop1_ratio': [3/4]}))

    # check zero denom
    ratios = neighborhood_analysis.compute_cell_ratios(
        cell_neighbors_mat, ['cell1'], ['cell3'], ['fov1'])
    assert ratios.equals(pd.DataFrame({'fov': 'fov1', 'pop1_pop2_ratio': [np.nan],
                                       'pop2_pop1_ratio': [np.nan]}))


def test_compute_mixing_score():
    cell_neighbors_mat = pd.DataFrame({
        settings.FOV_ID: ['fov1', 'fov1', 'fov1', 'fov1', 'fov1', 'fov1', 'fov1'],
        settings.CELL_LABEL: list(range(1, 8)),
        settings.CELL_TYPE: ['cell1', 'cell2', 'cell1', 'cell1', 'cell2', 'cell2', 'cell3'],
        'cell1': [1, 0, 2, 2, 1, 2, 0],
        'cell2': [1, 2, 1, 1, 2, 2, 0],
        'cell3': [0, 0, 0, 0, 0, 0, 1],
        'cell4': [0, 0, 0, 0, 0, 0, 0]
    })

    # check cell type validation
    with pytest.raises(ValueError, match='The following cell types were included in both '
                                         'the target and reference populations'):
        neighborhood_analysis.compute_mixing_score(
            cell_neighbors_mat, 'fov1', target_cells=['cell1'], reference_cells=['cell1'],
            mixing_type='homogeneous')

    with pytest.raises(ValueError, match='Not all values given in list provided column'):
        neighborhood_analysis.compute_mixing_score(
            cell_neighbors_mat, 'fov1', target_cells=['cell1'], reference_cells=['cell2'],
            mixing_type='homogeneous', cell_col='bad_column')
    with pytest.raises(ValueError, match='Please provide a valid mixing_type'):
        neighborhood_analysis.compute_mixing_score(
            cell_neighbors_mat, 'fov1', target_cells=['cell1'], reference_cells=['cell2'],
            mixing_type='bad')

    # check that extra cell type is ignored
    score = neighborhood_analysis.compute_mixing_score(
        cell_neighbors_mat, 'fov1', target_cells=['cell1', 'cell3', 'cell_not_in_fov'],
        reference_cells=['cell2'], mixing_type='homogeneous')
    assert score == 3 / 12

    # test homogeneous mixing
    score = neighborhood_analysis.compute_mixing_score(
        cell_neighbors_mat, 'fov1', target_cells=['cell1', 'cell3'], reference_cells=['cell2'],
        mixing_type='homogeneous')
    assert score == 3/12

    # test percent mixing
    score = neighborhood_analysis.compute_mixing_score(
        cell_neighbors_mat, 'fov1', target_cells=['cell1', 'cell3'], reference_cells=['cell2'],
        mixing_type='percent')
    assert score == 3 / 9

    # test ratio threshold
    cold_score = neighborhood_analysis.compute_mixing_score(
        cell_neighbors_mat, 'fov1', target_cells=['cell1'], reference_cells=['cell2'],
        ratio_threshold=0.5, mixing_type='homogeneous')
    assert math.isnan(cold_score)

    # test cell count threshold
    cold_score = neighborhood_analysis.compute_mixing_score(
        cell_neighbors_mat, 'fov1', target_cells=['cell1'], reference_cells=['cell2'],
        cell_count_thresh=5, mixing_type='homogeneous')
    assert math.isnan(cold_score)

    # check zero cells denominator
    cold_score = neighborhood_analysis.compute_mixing_score(
        cell_neighbors_mat, 'fov1', target_cells=['cell4'], reference_cells=['cell2'],
        cell_count_thresh=0, mixing_type='homogeneous')
    assert math.isnan(cold_score)
