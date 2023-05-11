import os
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from pytest_mock import MockerFixture

import ark.settings as settings
from ark.analysis import phenotype_by_neighbour_analysis


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


def test_calculate_median_distance_to_cell_type():
    # create cell data frame
    celldf = generate_test_celldf()

    # create distance matrix
    distdf = generate_test_distance_matrix()

    # test calculate distance to cell type
    cancer_dist = phenotype_by_neighbour_analysis.calculate_median_distance_to_cell_type(
        celldf, distdf, cell_cluster='Cancer', k=2)

    # check that distances are close to expected
    actual_dist = [0.56, 0.07, 0.295, 0.51, 0.685]
    assert np.all(np.isclose(cancer_dist, actual_dist))


def test_calculate_median_distance_to_all_cell_types():
    # create cell data frame
    celldf = generate_test_celldf()

    # create distance matrix
    distdf = generate_test_distance_matrix()

    # test calculate distance to all cell types
    cancer_dist = phenotype_by_neighbour_analysis.calculate_median_distance_to_all_cell_types(
        celldf, distdf, k=2)

    # set expected results
    actual_dist = pd.DataFrame({
        'Cancer': [0.560, 0.070, 0.295, 0.510, 0.685],
        'Immune': [0.745, 0.575, 0.760, 0.745, 0.225]
    })
    assert np.all(np.isclose(cancer_dist, actual_dist))


def test_cell_neighbor_distance_analysis(mocker: MockerFixture, ):
    mocker.patch('xarray.xr.load_dataarray', generate_test_distance_matrix())

    cell_table = pd.concat([generate_test_celldf('fov1'), generate_test_celldf('fov2')])

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, 'neighbor_distances.csv')

        cell_dists = phenotype_by_neighbour_analysis.cell_neighbor_distance_analysis(
            cell_table, temp_dir, save_path, k=2)

        # check all cells in dataframe
        assert cell_dists.shape[0] == cell_table.shape[0]

        # check columns are correct
        assert np.unique(cell_dists.settings.CELL_TYPE) in cell_dists.columns
        assert cell_dists.columns[0:3] == [settings.FOV_ID, settings.CELL_LABEL,
                                           settings.CELL_TYPE]
        # check file is saved
        assert os.path.exists(save_path)
