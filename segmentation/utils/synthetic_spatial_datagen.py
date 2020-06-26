import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import os


def generate_labels(num_A=100, num_B=100, num_C=100):
	"""
	This function will generate a set of associated labels for each cell centroid.

	A helper function to get_random_dist_matrix.

	Return value will be a list of values associated with AB and AC, as well as 
	a dictionary of mappings of the type cell_num: cell_label.

	Cell label will be either 'AB' or 'AC'

	Args:
		num_A: the number of A cells we wish to generate. Default 100.
		num_B: the number of B cells we wish to generate. Default 100.
		num_C: the number of C cells we wish to generate. Default 100.
	"""

	# Generate a range of numbers the same length as the total of A, B, and C labels desired
	# And select a random set of indices to identify as A cells
	num_range = np.arange(num_A + num_B + num_C)
	a_indices = np.random.choice(num_range.size, num_A, replace=False)

	# Get a indices
	a_values = num_range[a_indices]

	# From remaining B and C cells, select a random set of indices for B cells
	non_a_indices = num_range[~a_indices]
	b_indices = np.random.choice(non_a_indices, num_B, replace=False)

	# Get b indices, set the remainder as c indices
	b_indices = non_a_indices[b_indices]
	c_indices = non_a_indices[c_indices]

	a_dict = dict([(a, 'A') for a in a_values])
	b_dict = dict([(b, 'B') for b in b_values])
	c_dict = dict([(c, 'C') for c in c_values])

	label_dict = {**a_dict, **b_dict, **c_dict}

	return a_indices, b_indices, c_indices, label_dict


def get_random_dist_matrix(num_A=100, num_B=100, num_C=100, distr_AB=None, distr_AC=None):
	"""
	This function will return a random dist matrix such that the distance between cells
	of types A and B are overall larger than the distance between cells of types A and C

	Each row and column representing a cell.
	Rows represetnt 
	We generate the points using Gaussian distributions
	Ideally, the parameters for A to B distances will be set such that they produce a lower range of values
	than A to C distances.

	Will return a random distance matrix as well as the dictionary of associated cell: label IDs
	The above is generated from the generate_labels function

	Args:
		num_A: the number of A cells we wish to generate. Default 100
		num_B: the number of B cells we wish to generate. Default 100
		num_C: the number of C cells we wish to generate. Default 100
		distr_AB: if specified, will be a dict listing the mean and variance of the Gaussian distribution
			we wish to generate numbers from. If None, use the default values.
		distr_AC: similar to dist_AB. Default will have a higher mean value.
	"""

	# generate a list of A, B, and C cells
	labels_a, labels_b, labels_c, dict_labels = generate_labels(num_A, num_B, num_C)

	# initialize the distance matrix
	sample_dist_mat = np.zeros((num_A + num_B + num_C, num_A + num_B + num_C))

	# set the mean and variance of the Gaussian distributions of both AB and AC distances
	if distr_AB = None:
		mean_ab = 100
		var_ab = 1
	else:
		mean_ab = distr_AB['mean']
		var_ab = distr_AB['var']

	if distr_AC = None:
		mean_ac = 10
		var_ac = 1
	else:
		mean_ac = distr_AC['mean']
		var_ac = distr_AC['var']

	# generate a random numpy matrix of Gaussian values from specified distribution
	# and assign to corresponding labels_a, labels_b values
	random_ab = np.random.normal(mean_ab, var_ab, (num_A, num_B))
	# random_ab = (random_ab + random_ab.T) / 2
	sample_dist_mat[labels_a, labels_b] = random_ab
	sample_dist_mat[labels_b, labels_a] = random_ab.T

	# assert that the created submatrix is symmetric
	assert np.alclose(sample_dist_mat[labels_a, labels_b], sample_dist_mat[labels_a, labels_b].T, rtol=1e-05, atol=1e-08)

	# follow the same steps for labels_a and labels_c
	random_ac = np.random.normal(mean_ac, var_ac, (num_A, num_C))
	random_ac = (random_ac + random_ac.T) / 2
	sample_dist_mat[labels_a, labels_c] = random_ac

	assert np.alclose(sample_dist_mat[labels_a, labels_c], sample_dist_mat[labels_a, labels_c].T, rtol=1e-05, atol=1e-08)

	# we don't care about a-a, b-b, c-c, or b-c distances, so we just return the matrix along with the labels
	return sample_dist_mat, dict_labels