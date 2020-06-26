import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import os


def generate_labels(num_AB=1000, num_AC=1000):
	"""
	This function will generate a set of associated labels for each cell centroid.
	A helper function to get_random_dist_matrix.
	Return value will be a dictionary of mappings of the type cell_num: cell_label.
	Cell label will be either 'AB' or 'AC'

	Args:
		num_AB: the number of AB distances we wish to generate.
		num_AC: the number of AC distances we wish to generate.
	"""

	# Generate a range of numbers the same length as the total of AB and AC labels desired
	# And select a random set of indices to identify as AB connections
	num_range = np.arange(num_AB + num_AC)
	ab_indices = np.random.choice(num_range.size, num_AB, replace=False)

	# Generate a 
	ab_values = num_range[ab_indices]
	ac_values = num_range[~ab_indices]

	ab_dict = dict([(ab, 'AB') for ab in ab_values])
	ac_dict = dict([(ac, 'AC') for ac in ac_values])

	label_dict = {**ab_dict, **ac_dict}

	return label_dict


def get_random_dist_matrix(num_AB=1000, num_AC=1000, distr_AB=None, distr_AC=None):
	"""
	This function will return a random dist matrix such that the distance between cells
	of types A and B are overall larger than the distance between cells of types A and C

	Each row and column representing a type of cell.
	We generate the points using a Gaussian distribution for AB and AC.
	Ideally, the parameters for AB will be set such that they produce a lower range of values
	than AC.

	Will return a random distance matrix as well as the dictionary of associated cell: label IDs
	The above is generated from the generate_labels function

	Args:
		num_AB: the number of AB distances we wish to generate. Default 1000
		num_AC: the number of AC distances we wish to generate. Default 1000
		distr_AB: if specified, will be a tuple the mean and variance of the Gaussian distribution
			we wish to generate numbers from. If None, use the default values.
		distr_AC: similar to dist_AB. Default will have a higher mean value.
	"""

	labels = generate_labels(num_AB, num_AC)

	return None