import os
from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm
from alpineer import io_utils, misc_utils

import ark.settings as settings


def shannon_diversity(proportions):
    """ Calculates the shannon diversity index for the provided proportions of a community
    Args:
        proportions (np.array):
            the proportions of each individual group

    Returns:
        float:
            the diversity of neighborhood
    """

    prop_index = proportions > 0
    return -np.sum(proportions[prop_index] * np.log2(proportions[prop_index]))


def compute_neighborhood_diversity(neighborhood_mat, cell_type_col):
    """ Generates a diversity score for each cell using the neighborhood matrix
    Args:
        neighborhood_mat (pd.DataFrame):
            the frequency neighbors matrix
        cell_type_col (string):
            the specific name of the cell type column the matrix represents

    Returns:
        pd.DataFrame:
            contains the fov, label, cell_type, and diversity_cell_type values for each cell
    """

    misc_utils.verify_in_list(cell_type_column=cell_type_col,
                              neighbor_matrix_columns=neighborhood_mat.columns)

    # check input values
    neighborhood_mat_values = np.array(neighborhood_mat.drop(
        columns=[settings.FOV_ID, settings.CELL_LABEL, cell_type_col]))
    if (neighborhood_mat_values > 1).any():
        raise ValueError("Input must be frequency values.")

    diversity_data = []
    fov_list = np.unique(neighborhood_mat[settings.FOV_ID])
    with tqdm(total=len(fov_list), desc="Calculate Neighborhood Diversity", unit="FOVs") \
            as diversity_progress:
        for fov in fov_list:
            diversity_progress.set_postfix(FOV=fov)

            fov_neighborhoods = neighborhood_mat[neighborhood_mat[settings.FOV_ID] == fov]

            diversity_scores = []
            cells = fov_neighborhoods[settings.CELL_LABEL]
            for label in cells:
                # retrieve an array of only the neighbor frequencies for the cell
                neighbor_freqs = \
                    fov_neighborhoods[fov_neighborhoods[settings.CELL_LABEL] == label].drop(
                        columns=[settings.FOV_ID, settings.CELL_LABEL, cell_type_col]).values[0]

                diversity_scores.append(shannon_diversity(neighbor_freqs))

            # combine the data for cells in the image
            fov_data = pd.DataFrame({
                settings.FOV_ID: [fov] * len(cells),
                settings.CELL_LABEL: cells,
                cell_type_col: fov_neighborhoods[cell_type_col],
                f'diversity_{cell_type_col}': diversity_scores
            })
            diversity_data.append(fov_data)

            diversity_progress.update(1)

    # dataframe containing all fovs
    diversity_data = pd.concat(diversity_data)

    return diversity_data


def generate_neighborhood_diversity_analysis(neighbors_mat_dir, pixel_radius, cell_type_columns):
    """ Generates a diversity score for each cell using the neighborhood matrix
    Args:
        neighbors_mat_dir (str):
            directory containing the neighbors matrices
        pixel_radius (int):
            radius used to define the neighbors of each cell
        cell_type_columns (list):
            list of cell cluster columns to read in neighbors matrices for

    Returns:
        pd.DataFrame:
            contains diversity data calculated at each specified cell cluster level
    """

    freqs_mat_paths = [os.path.join(neighbors_mat_dir,
                                    f"neighborhood_freqs-{cell_type_col}_radius{pixel_radius}.csv")
                       for cell_type_col in cell_type_columns]
    io_utils.validate_paths(freqs_mat_paths)

    diversity_data = []
    for cell_type_col, freqs_path in zip(cell_type_columns, freqs_mat_paths):
        neighbor_freqs = pd.read_csv(freqs_path)
        diversity_data.append(compute_neighborhood_diversity(neighbor_freqs, cell_type_col))

    all_diversity_data = reduce(
        lambda left, right: pd.merge(left, right, on=[settings.FOV_ID, settings.CELL_LABEL]),
        diversity_data)

    return all_diversity_data


def calculate_mean_distance_to_cell_type(
        cell_table, dist_xr, cell_cluster, k, cell_type_col=settings.CELL_TYPE,
        cell_label_col=settings.CELL_LABEL):
    """Function to calculate mean distance of all cells to a specified cell type
    Args:
        cell_table (pd.DataFrame):
            Dataframe containing all cells and their cell type
        dist_xr (xr.array):
            Cell by cell distances for all cells
        cell_cluster (str):
            Cell cluster to calculate distance to
        k (int):
            Number of nearest neighbours
        cell_type_col (str):
            column with the cell phenotype
        cell_label_col (str):
            column with the cell labels

    Returns:
        np.array:
            mean distances for each cell to the cluster cells
    """

    # get cell ids for all cells of specific cluster
    cluster_labels = cell_table.loc[cell_table[cell_type_col] == cell_cluster, cell_label_col]

    # get all cells that match specified cell cluster
    dist_xr = dist_xr.loc[:, dist_xr.dim_1.isin(cluster_labels)]

    # keep the closest k values, not including itself
    dist_xr = dist_xr.where(dist_xr > 0)
    if dist_xr.shape[1] < k:
        # image must contain at least k cell_cluster cells to receive an average dist
        return [np.nan] * len(dist_xr)

    sorted_dist = np.sort(dist_xr.values, axis=1)
    sorted_dist = sorted_dist[:, :k]

    # take the mean
    mean_dists = sorted_dist.mean(axis=1)

    return mean_dists


def calculate_mean_distance_to_all_cell_types(
        cell_table, dist_xr, k, cell_type_col=settings.CELL_TYPE,
        cell_label_col=settings.CELL_LABEL):
    """Wrapper function to calculate mean distance of all cells against all cell types
    Args:
        cell_table (pd.DataFrame):
            Dataframe containing all cells and their cell type
        dist_xr (xr.array):
            Cell by cell distances for all cells
        k (int):
            Number of nearest neighbours
        cell_type_col (str):
            column with the cell phenotype
        cell_label_col (str):
            column with the cell labels

    Returns:
        pd.DataFrame:
            average distances
    """

    # get all cell clusters in cell table
    all_clusters = np.unique(cell_table[cell_type_col])

    # call calculate_mean_distance_to_cell_type for all cell clusters
    avg_dists = pd.DataFrame(index=cell_table.index.values, columns=all_clusters, dtype=np.float64)
    for cell_cluster in all_clusters:
        avg_dists.loc[:, cell_cluster] = calculate_mean_distance_to_cell_type(
            cell_table, dist_xr, cell_cluster, k, cell_type_col, cell_label_col)

    return avg_dists


def generate_cell_distance_analysis(
        cell_table, dist_mat_dir, save_path, k, cell_type_col=settings.CELL_TYPE,
        fov_col=settings.FOV_ID, cell_label_col=settings.CELL_LABEL):
    """ Creates a dataframe containing the average distance between a cell and other cells of each
    phenotype, based on the specified cell_type_col.
    Args:
        cell_table (pd.DataFrame):
            dataframe containing all cells and their cell type
        dist_mat_dir (str):
            path to directory containing the distance matrix files
        save_path (str):
            path where to save the results to
        k (int):
            Number of nearest neighbours
        fov_col (str):
            column containing the image name
        cell_type_col (str):
            column with the cell phenotype
        cell_label_col (str):
            column with the cell labels
    """

    io_utils.validate_paths(dist_mat_dir)
    fov_list = np.unique(cell_table[fov_col])

    cell_dists = []
    with tqdm(total=len(fov_list), desc="Calculate Average Distances", unit="FOVs") \
            as distance_progress:
        for fov in fov_list:
            distance_progress.set_postfix(FOV=fov)

            fov_cell_table = cell_table[cell_table[fov_col] == fov]
            fov_dist_xr = xr.load_dataarray(os.path.join(dist_mat_dir, str(fov) + '_dist_mat.xr'))

            # get the average distances between cell types
            fov_cell_dists = calculate_mean_distance_to_all_cell_types(
                fov_cell_table, fov_dist_xr, k, cell_type_col, cell_label_col)

            # add the fov name and cell phenotypes to the dataframe
            fov_cell_dists.insert(0, fov_col, fov)
            fov_cell_dists.insert(1, cell_label_col, fov_cell_table[cell_label_col])
            fov_cell_dists.insert(2, cell_type_col, fov_cell_table[cell_type_col])
            cell_dists.append(fov_cell_dists)

            distance_progress.update(1)

    # combine data for all fovs and save to csv
    all_cell_dists = pd.concat(cell_dists)
    all_cell_dists.to_csv(save_path, index=False)

    return all_cell_dists
