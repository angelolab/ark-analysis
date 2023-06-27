import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest_mock import MockerFixture

import ark.settings as settings
from ark.analysis import cell_neighborhood_stats


def test_shannon_diversity():

    props1 = np.array([0, 0, 1])
    assert cell_neighborhood_stats.shannon_diversity(props1) == 0

    props2 = np.array([0, 0.25, 0, 0.15, 0.6])
    assert cell_neighborhood_stats.shannon_diversity(props2) == \
           -(0.25*np.log2(0.25)+0.15*np.log2(0.15)+0.6*np.log2(0.6))


def test_compute_neighborhood_diversity():

    neighbor_counts = pd.DataFrame({
        settings.FOV_ID: ['fov1', 'fov1'],
        settings.CELL_LABEL: list(range(1, 3)),
        settings.CELL_TYPE: ['cell1', 'cell2'],
        'cell1': [1, 0],
        'cell2': [1, 2],
    })

    # counts matrix instead of freq should raise an error
    with pytest.raises(ValueError, match="Input must be frequency values."):
        cell_neighborhood_stats.compute_neighborhood_diversity(neighbor_counts, settings.CELL_TYPE)

    neighbor_freqs = pd.DataFrame({
        settings.FOV_ID: ['fov1', 'fov1', 'fov1', 'fov2'],
        settings.CELL_LABEL: [1, 2, 3, 1],
        settings.CELL_TYPE: ['cell1', 'cell2', 'cell2', 'cell1'],
        'cell1': [0.4, 0, 0.5, 0.3],
        'cell2': [0.3, 1, 0.5, 0],
        'cell3': [0.3, 0, 0, 0.7]
    })

    diversity_data = cell_neighborhood_stats.compute_neighborhood_diversity(
        neighbor_freqs, settings.CELL_TYPE)

    # check for shannon diversity column
    assert f'diversity_{settings.CELL_TYPE}' in diversity_data.columns

    # check every cell is in the new dataframe
    assert diversity_data.shape[0] == neighbor_freqs.shape[0]

    # check for high vs low diversity
    assert diversity_data[f'diversity_{settings.CELL_TYPE}'].max() == \
           diversity_data[f'diversity_{settings.CELL_TYPE}'][0]
    assert diversity_data[f'diversity_{settings.CELL_TYPE}'].min() == \
           diversity_data[f'diversity_{settings.CELL_TYPE}'][1]


def test_generate_neighborhood_diversity_analysis():

    with tempfile.TemporaryDirectory() as temp_dir:
        radius = 50
        cell_type_cols = ['cell_meta_cluster', 'cell_cluster']
        for col in cell_type_cols:
            cell1_freqs = [0.7, 0, 0.5, 0.2]
            cell2_freqs = [0.3, 1, 0.5, 0.8]

            if col == cell_type_cols[1]:
                cell1_freqs = [1, 0.9, 0.6, 0.35]
                cell2_freqs = [0, 0.1, 0.4, 0.55]

            neighbor_freqs = pd.DataFrame({
                settings.FOV_ID: ['fov1', 'fov1', 'fov1', 'fov2'],
                settings.CELL_LABEL: [1, 2, 3, 1],
                col: ['cell1', 'cell2', 'cell2', 'cell1'],
                'cell1': cell1_freqs,
                'cell2': cell2_freqs,
            })
            neighbor_freqs.to_csv(os.path.join(
                temp_dir, f"neighborhood_freqs-{col}_radius{radius}.csv"), index=False)

        # test success
        all_data = cell_neighborhood_stats.generate_neighborhood_diversity_analysis(
           neighbors_mat_dir=temp_dir, pixel_radius=radius, cell_type_columns=cell_type_cols)

        # check for multiple cell cluster columns
        assert np.isin(cell_type_cols, all_data.columns).all()

        diversity_columns = [f"diversity_{col}" for col in cell_type_cols]
        assert np.isin(diversity_columns, all_data.columns).all()

        # check every cell is in the new dataframe
        assert all_data.shape[0] == all_data.shape[0]

        # check score calculation by cell cluster
        for index, row in all_data.iterrows():
            assert row.diversity_cell_meta_cluster != row.diversity_cell_cluster


def generate_test_celldf(fov_name='fov1'):
    # create cell data frame
    celldf = pd.DataFrame({
        'ECAD': [0.01, 0.003, 0.009, 0.001, 0.01],
        'CD45': [0.001, 0.01, 0.01, 0.01, 0.004],
        'CD20': [0.002, 0.003, 0.003, 0.009, 0.001],
        'FOXP3': [0.001, 0.002, 0.01, 0.001, 0.003],
        settings.CELL_TYPE: ['Cancer', 'Immune', 'Immune', 'Immune', 'Cancer'],
        settings.CELL_LABEL: range(5),
        settings.FOV_ID: [fov_name] * 5
    })
    return celldf


def generate_test_distance_matrix():
    # create distance matrix
    disttmp = np.array([
        [1, 0.97, 0.79, 0.70, 0.12],
        [0.13, 1, 0.57, 0.58, 0.01],
        [0.58, 0.94, 1, 0.58, 0.01],
        [0.44, 0.76, 0.73, 1, 0.58],
        [0.37, 0.77, 0.07, 0.38, 1]
    ])
    distdf = xr.DataArray(
        disttmp,
        [("dim_0", range(5)), ("dim_1", range(5))]
    )
    return distdf


def test_calculate_mean_distance_to_cell_type():
    # create cell data frame
    celldf = generate_test_celldf()

    # create distance matrix
    distdf = generate_test_distance_matrix()

    # test calculate distance to cell type
    cancer_dist = cell_neighborhood_stats.calculate_mean_distance_to_cell_type(
        celldf, distdf, cell_cluster='Cancer', k=2)

    # check that distances are close to expected
    actual_dist = [0.56, 0.07, 0.295, 0.51, 0.685]
    assert np.all(np.isclose(cancer_dist, actual_dist))

    # test for insufficient number of Cancer cells in the image
    cancer_dist = cell_neighborhood_stats.calculate_mean_distance_to_cell_type(
        celldf, distdf, cell_cluster='Cancer', k=3)

    # check for nan values
    assert np.isnan(cancer_dist).all()


def test_calculate_mean_distance_to_all_cell_types():
    # create cell data frame
    celldf = generate_test_celldf()

    # create distance matrix
    distdf = generate_test_distance_matrix()

    # test calculate distance to all cell types
    cancer_dist = cell_neighborhood_stats.calculate_mean_distance_to_all_cell_types(
        celldf, distdf, k=2)

    # set expected results
    actual_dist = pd.DataFrame({
        'Cancer': [0.560, 0.070, 0.295, 0.510, 0.685],
        'Immune': [0.745, 0.575, 0.760, 0.745, 0.225]
    })
    assert pd.testing.assert_frame_equal(cancer_dist, actual_dist, check_exact=False) is None


def test_generate_cell_distance_analysis(mocker: MockerFixture, ):
    mocker.patch('xarray.load_dataarray', return_value=generate_test_distance_matrix())

    cell_table = pd.concat([generate_test_celldf('fov1'), generate_test_celldf('fov2')])

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, 'neighbor_distances.csv')

        cell_dists = cell_neighborhood_stats.generate_cell_distance_analysis(
            cell_table, temp_dir, save_path, k=2)

        # check all cells in dataframe
        assert cell_dists.shape[0] == cell_table.shape[0]

        # check columns are correct
        assert np.isin(np.unique(cell_dists[settings.CELL_TYPE]), cell_dists.columns).all()
        assert (cell_dists.columns[0:3] == [settings.FOV_ID, settings.CELL_LABEL,
                                            settings.CELL_TYPE]).all()
        # check file is saved
        assert os.path.exists(save_path)
