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
	Cell label will be either 'A', 'B', or 'C'

	Args:
		num_AB: the number of AB distances we wish to generate.
		num_AC: the number of AC distances we wish to generate.
	"""

def get_random_dist_matrix(num_AB=1000, num_AC=1000, dist_AB=None, dist_AC=None):
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
		dist_AB: if specified, will be a tuple the mean and variance of the Gaussian distribution
			we wish to generate numbers from. If None, use the default values.
		dist_AC: similar to dist_AB. Default will have a higher mean value.
	"""