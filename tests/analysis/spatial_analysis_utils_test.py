import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import test_utils
from anndata import read_zarr

import ark.settings as settings
from ark.analysis import spatial_analysis_utils


def test_calc_dist_matrix():
    with tempfile.TemporaryDirectory() as base_dir:
        # generate sample data
        anndata_dir = os.path.join(base_dir, "anndata")
        os.makedirs(anndata_dir)

        centroids_8 = [(0, 0), (3, 4), (6, 0), (6, 8)]
        centroids_9 = [(6, 8), (0, 16), (0, 0), (12, 0)]
        for fov_num in [8, 9]:
            adata = test_utils.generate_anndata_table(rng=np.random.default_rng(), n_obs=4,
                                                      n_vars=2, fov_id=fov_num, obs_properties=0,
                                                      obs_categorical_properties=0)
            centroid_list = centroids_8 if fov_num == 8 else centroids_9

            for i in adata.obs.label:
                adata.obsm['spatial'].centroid_y[f"cell_{i}"] = centroid_list[i][0]
                adata.obsm['spatial'].centroid_x[f"cell_{i}"] = centroid_list[i][1]
            adata.write_zarr(os.path.join(anndata_dir, f"fov_{fov_num}.zarr"))

        # generate the distance matrices
        spatial_analysis_utils.calc_dist_matrix(anndata_dir)

        # assert the fov8 and fov9 .xr files exist
        for fov_num in [8, 9]:
            assert os.path.exists(
                os.path.join(anndata_dir, f"fov_{fov_num}.zarr", "obsp", "distances"))

        # verify fov8, use is close to prevent float tolerance errors
        fov8_adata = read_zarr(os.path.join(anndata_dir, "fov_8.zarr")).obsp["distances"]
        actual8_data = np.array([
            [0, 5, 6, 10],
            [5, 0, 5, 5],
            [6, 5, 0, 8],
            [10, 5, 8, 0]
        ])
        assert np.all(np.isclose(fov8_adata, actual8_data))

        # verify fov9, use isclose to prevent float tolerance errors
        fov9_adata = read_zarr(os.path.join(anndata_dir, "fov_9.zarr")).obsp["distances"]
        actual9_data = np.array([
            [0, 10, 10, 10],
            [10, 0, 16, 20],
            [10, 16, 0, 12],
            [10, 20, 12, 0]
        ])
        assert np.all(np.isclose(fov9_adata, actual9_data))


def test_append_distance_features_to_dataset():
    all_data, dist_mat = test_utils._make_dist_exp_mats_spatial_utils_test()

    feat_dist = 300

    all_data['dist_whole_cell'] = feat_dist * np.ones(all_data.shape[0])

    num_labels = max(all_data[settings.CELL_LABEL].unique())
    num_cell_types = max(list(all_data[settings.CELL_TYPE].astype("category").cat.codes)) + 1

    all_data, dist_mat = spatial_analysis_utils.append_distance_features_to_dataset(
        'fov8', dist_mat, all_data, ['dist_whole_cell']
    )

    appended_cell_row = all_data.iloc[-1, :][[
        settings.CELL_LABEL,
        settings.FOV_ID,
        settings.CELL_TYPE,
        settings.CELL_TYPE_NUM,
    ]]
    pd.testing.assert_series_equal(appended_cell_row, pd.Series({
        settings.CELL_LABEL: num_labels + 1,
        settings.FOV_ID: 'fov8',
        settings.CELL_TYPE: 'dist_whole_cell',
        settings.CELL_TYPE_NUM: num_cell_types + 1,
    }), check_names=False)

    dist_mat_new_row = dist_mat.values[-1, :]
    dist_mat_new_col = dist_mat.values[:, -1]

    expected = feat_dist * np.ones(all_data.shape[0])
    expected[-1] = np.nan

    np.testing.assert_equal(dist_mat_new_row, expected)
    np.testing.assert_equal(dist_mat_new_col, expected)


def test_get_pos_cell_labels_channel():
    all_data, _ = test_utils._make_dist_exp_mats_spatial_utils_test()
    example_thresholds = test_utils._make_threshold_mat(in_utils=True)

    excluded_channels = [0, 13, 22]

    # Subsets the expression matrix to only have channel columns
    channel_start = np.where(all_data.columns == settings.PRE_CHANNEL_COL)[0][0] + 1
    channel_end = np.where(all_data.columns == settings.POST_CHANNEL_COL)[0][0]

    fov_channel_data = all_data.iloc[:, channel_start:channel_end]
    fov_channel_data = fov_channel_data.drop(fov_channel_data.columns[excluded_channels], axis=1)

    thresh_vec = example_thresholds.iloc[0:20, 1]

    cell_labels = all_data.loc[:, settings.CELL_LABEL]

    pos_cell_labels = spatial_analysis_utils.get_pos_cell_labels_channel(
        thresh_vec.iloc[0], fov_channel_data, cell_labels, fov_channel_data.columns[0])

    assert len(pos_cell_labels) == 4


def test_get_pos_cell_labels_cluster():
    all_data, _ = test_utils._make_dist_exp_mats_spatial_utils_test()
    all_data[settings.CELL_TYPE_NUM] = list(all_data[settings.CELL_TYPE].
                                            astype('category').cat.codes)
    excluded_channels = [0, 13, 22]

    # Subsets the expression matrix to only have channel columns
    channel_start = np.where(all_data.columns == settings.PRE_CHANNEL_COL)[0][0] + 1
    channel_end = np.where(all_data.columns == settings.POST_CHANNEL_COL)[0][0]

    fov_channel_data = all_data.iloc[:, list(range(channel_start, channel_end + 1)) + [32]]
    fov_channel_data = fov_channel_data.drop(fov_channel_data.columns[excluded_channels], axis=1)

    cluster_ids = all_data.loc[:, settings.CELL_TYPE_NUM].drop_duplicates()

    pos_cell_labels = spatial_analysis_utils.get_pos_cell_labels_cluster(
        cluster_ids.iloc[0], fov_channel_data, settings.CELL_LABEL, settings.CELL_TYPE_NUM)

    assert len(pos_cell_labels) == 4


def test_compute_close_cell_num():
    # Test the closenum function
    all_data, example_dist_mat = test_utils._make_dist_exp_mats_spatial_utils_test()
    example_thresholds = test_utils._make_threshold_mat(in_utils=True)

    all_data[settings.CELL_TYPE_NUM] = list(all_data[settings.CELL_TYPE].
                                            astype('category').cat.codes)

    excluded_channels = [0, 13, 22]

    # Subsets the expression matrix to only have channel columns
    channel_start = np.where(all_data.columns == settings.PRE_CHANNEL_COL)[0][0] + 1
    channel_end = np.where(all_data.columns == settings.POST_CHANNEL_COL)[0][0]

    fov_channel_data = all_data.iloc[:, channel_start:channel_end]
    fov_channel_data = fov_channel_data.drop(fov_channel_data.columns[excluded_channels], axis=1)

    # Subsetting threshold matrix to only include column with threshold values
    thresh_vec = example_thresholds.iloc[0:20, 1].values

    # not taking into account mark1labels_per_id return value
    example_closenum, m1, _ = spatial_analysis_utils.compute_close_cell_num(
        dist_mat=example_dist_mat, dist_lim=100, analysis_type="channel",
        current_fov_data=all_data, current_fov_channel_data=fov_channel_data,
        thresh_vec=thresh_vec)

    assert (example_closenum[:2, :2] == 12).all()
    assert (example_closenum[3:5, 3:5] == 20).all()
    assert (example_closenum[5:7, 5:7] == 0).all()

    # Now test indexing with cell labels by removing a cell label from the expression matrix but
    # not the distance matrix
    all_data = all_data.drop(3, axis=0)
    fov_channel_data = fov_channel_data.drop(3, axis=0)

    # Subsetting threshold matrix to only include column with threshold values
    thresh_vec = example_thresholds.iloc[0:20, 1].values

    example_closenum, m1, _ = spatial_analysis_utils.compute_close_cell_num(
        dist_mat=example_dist_mat, dist_lim=100, analysis_type="channel",
        current_fov_data=all_data, current_fov_channel_data=fov_channel_data,
        thresh_vec=thresh_vec)

    assert (example_closenum[:2, :2] == 6).all()
    assert (example_closenum[3:5, 3:5] == 20).all()
    assert (example_closenum[5:7, 5:7] == 0).all()

    # now, test for cluster enrichment
    all_data, example_dist_mat = test_utils._make_dist_exp_mats_spatial_utils_test()
    all_data[settings.CELL_TYPE_NUM] = list(all_data[settings.CELL_TYPE].
                                            astype('category').cat.codes)
    cluster_ids = all_data.loc[:, settings.CELL_TYPE_NUM].drop_duplicates().values

    example_closenum, m1, _ = spatial_analysis_utils.compute_close_cell_num(
        dist_mat=example_dist_mat, dist_lim=100, analysis_type="cluster",
        current_fov_data=all_data, cluster_ids=cluster_ids)

    assert example_closenum[0, 0] == 12
    assert example_closenum[1, 1] == 20
    assert example_closenum[2, 2] == 0


def test_compute_close_cell_num_random():
    data_markers, example_distmat = test_utils._make_dist_exp_mats_spatial_utils_test()

    marker_pos_labels = [
        data_markers[data_markers[settings.CELL_TYPE] == lineage][settings.CELL_LABEL]
        for lineage in data_markers[settings.CELL_TYPE].unique()
    ]

    # Generate random inputs to test shape
    marker_nums = [len(marker_labels) for marker_labels in marker_pos_labels]

    example_closenumrand = spatial_analysis_utils.compute_close_cell_num_random(
        marker_nums, marker_pos_labels, example_distmat, dist_lim=100, bootstrap_num=100
    )

    assert example_closenumrand.shape == (len(marker_nums), len(marker_nums), 100)

    # test asymmetry
    assert (example_closenumrand[0, 1, :] != example_closenumrand[1, 0, :]).any()

    # bad marker nums
    marker_nums[0] = example_distmat.shape[0] + 1

    with pytest.raises(ValueError):
        example_closenumrand = spatial_analysis_utils.compute_close_cell_num_random(
            marker_nums, marker_pos_labels, example_distmat, dist_lim=100, bootstrap_num=100
        )


def test_calculate_enrichment_stats():
    # Positive enrichment

    # Generate random closenum matrix
    stats_cnp = np.zeros((20, 20))
    stats_cnp[:, :] = 80

    # Generate random closenumrand matrix, ensuring significant positive enrichment
    stats_cnrp = np.random.randint(1, 40, (20, 20, 100))

    stats_xr_pos = spatial_analysis_utils.calculate_enrichment_stats(stats_cnp, stats_cnrp)

    assert stats_xr_pos.loc["z", 0, 0] > 0
    assert stats_xr_pos.loc["p_pos", 0, 0] < .05

    # Negative enrichment

    # Generate random closenum matrix
    stats_cnn = np.zeros((20, 20))

    # Generate random closenumrand matrix, ensuring significant negative enrichment
    stats_cnrn = np.random.randint(40, 80, (20, 20, 100))

    stats_xr_neg = spatial_analysis_utils.calculate_enrichment_stats(stats_cnn, stats_cnrn)

    assert stats_xr_neg.loc["z", 0, 0] < 0
    assert stats_xr_neg.loc["p_neg", 0, 0] < .05

    # No enrichment

    # Generate random closenum matrix
    stats_cn = np.zeros((20, 20))
    stats_cn[:, :] = 80

    # Generate random closenumrand matrix, ensuring no enrichment
    stats_cnr = np.random.randint(78, 82, (20, 20, 100))

    stats_xr = spatial_analysis_utils.calculate_enrichment_stats(stats_cn, stats_cnr)

    assert abs(stats_xr.loc["z", 0, 0]) < 1
    assert stats_xr.loc["p_neg", 0, 0] > .05
    assert stats_xr.loc["p_pos", 0, 0] > .05


def test_compute_neighbor_counts():
    fov_col = settings.FOV_ID
    cluster_id_col = settings.CELL_TYPE_NUM
    cell_label_col = settings.CELL_LABEL
    cluster_name_col = settings.CELL_TYPE
    distlim = 100

    fov_data, dist_matrix = test_utils._make_dist_exp_mats_spatial_utils_test()
    fov_data[cluster_id_col] = list(fov_data[settings.CELL_TYPE].astype('category').cat.codes)

    cluster_names = fov_data[cluster_name_col].drop_duplicates()
    fov_data = fov_data[[fov_col, cell_label_col, cluster_id_col, cluster_name_col]]
    cluster_num = len(fov_data[cluster_id_col].drop_duplicates())

    cell_neighbor_counts = pd.DataFrame(np.zeros((fov_data.shape[0], cluster_num + 2)))

    cell_neighbor_counts[[0, 1]] = fov_data[[fov_col, cell_label_col]]

    # Rename the columns to match cell phenotypes
    cols = [fov_col, cell_label_col] + list(cluster_names)
    cell_neighbor_counts.columns = cols

    cell_neighbor_freqs = cell_neighbor_counts.copy(deep=True)

    # first test for self_neighbor is True
    counts, freqs = spatial_analysis_utils.compute_neighbor_counts(
        fov_data, dist_matrix, distlim, self_neighbor=True)

    # add to neighbor counts/freqs for only matched phenos between the fov and the whole dataset
    cell_neighbor_counts.loc[fov_data.index, cluster_names] = counts
    cell_neighbor_freqs.loc[fov_data.index, cluster_names] = freqs

    assert (cell_neighbor_counts.loc[:3, "Pheno1"] == 4).all()
    assert (cell_neighbor_counts.loc[4:8, "Pheno2"] == 5).all()
    assert (cell_neighbor_counts.loc[9, "Pheno3"] == 1).all()

    assert (cell_neighbor_freqs.loc[:3, "Pheno1"] == 1).all()
    assert (cell_neighbor_freqs.loc[4:8, "Pheno2"] == 1).all()
    assert (cell_neighbor_freqs.loc[9, "Pheno3"] == 1).all()

    # now test for self_neighbor is False, first reset values
    cell_neighbor_counts = pd.DataFrame(np.zeros((fov_data.shape[0], cluster_num + 2)))
    cell_neighbor_counts[[0, 1]] = fov_data[[fov_col, cell_label_col]]
    cell_neighbor_counts.columns = cols
    cell_neighbor_freqs = cell_neighbor_counts.copy(deep=True)

    counts, freqs = spatial_analysis_utils.compute_neighbor_counts(
        fov_data, dist_matrix, distlim, self_neighbor=False)

    cell_neighbor_counts.loc[fov_data.index, cluster_names] = counts
    cell_neighbor_freqs.loc[fov_data.index, cluster_names] = freqs

    assert (cell_neighbor_counts.loc[:3, "Pheno1"] == 3).all()
    assert (cell_neighbor_counts.loc[4:8, "Pheno2"] == 4).all()
    assert (cell_neighbor_counts.loc[9, "Pheno3"] == 0).all()

    assert (cell_neighbor_freqs.loc[:3, "Pheno1"] == 1).all()
    assert (cell_neighbor_freqs.loc[4:8, "Pheno2"] == 1).all()
    assert (cell_neighbor_freqs.loc[9, "Pheno3"] == 0).all()


def test_compute_kmeans_inertia():
    neighbor_mat = test_utils._make_neighborhood_matrix()[['feature1', 'feature2']]

    neighbor_inertia_stats = spatial_analysis_utils.compute_kmeans_inertia(
        neighbor_mat, min_k=3, max_k=6)

    # assert we have the right cluster_num values
    assert list(neighbor_inertia_stats.coords["cluster_num"].values) == list(range(3, 7))

    # assert k=6 produces the best inertia
    best_inertia_score = neighbor_inertia_stats.loc[6].values
    assert np.all(best_inertia_score <= neighbor_inertia_stats.values)


def test_compute_kmeans_silhouette():
    neighbor_mat = test_utils._make_neighborhood_matrix()[['feature1', 'feature2']]

    neighbor_silhouette_stats = spatial_analysis_utils.compute_kmeans_silhouette(
        neighbor_mat, min_k=3, max_k=6)

    # assert we have the right cluster_num values
    assert list(neighbor_silhouette_stats.coords["cluster_num"].values) == list(range(3, 7))

    # assert k=3 produces the best silhouette score
    best_silhouette_score = neighbor_silhouette_stats.loc[3].values
    assert np.all(best_silhouette_score >= neighbor_silhouette_stats.values)


def test_generate_cluster_labels():
    neighbor_mat = test_utils._make_neighborhood_matrix()[['feature1', 'feature2']]
    neighbor_cluster_labels = spatial_analysis_utils.generate_cluster_labels(neighbor_mat,
                                                                             cluster_num=3)

    assert len(np.unique(neighbor_cluster_labels) == 3)
