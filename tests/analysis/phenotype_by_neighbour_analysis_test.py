import numpy as np
import pandas as pd
import xarray as xr

import ark.settings as settings
from ark.analysis import phenotype_by_neighbour_analysis


def generate_test_celldf():
	# create cell data frame
	celldf = pd.DataFrame({
		'ECAD': [0.01, 0.003, 0.009, 0.001, 0.01],
		'CD45': [0.001, 0.01, 0.01, 0.01, 0.004],
		'CD20': [0.002, 0.003, 0.003, 0.009, 0.001],
		'FOXP3': [0.001, 0.002, 0.01, 0.001, 0.003],
		settings.CELL_TYPE: ['Cancer', 'Immune', 'Immune', 'Immune', 'Cancer'],
		settings.CELL_LABEL: range(5)
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
