import os
import numpy as np
import pandas as pd
import xarray as xr

import ark.settings as settings
from alpineer import io_utils


def calculate_median_distance_to_cell_type(
        cell_table, dist_xr, cell_cluster, k, cell_type_col=settings.CELL_TYPE,
        cell_label_col=settings.CELL_LABEL):
    """Function to calculate median distance of all cells to a specified cell type
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
    j = cell_table.loc[cell_table[cell_type_col] == cell_cluster, cell_label_col]

    # get all cells that match specified cell cluster
    dist_xr = dist_xr.loc[:, dist_xr.dim_1.isin(j)]

    # keep the closest k values, not included itself
    dist_xr = dist_xr.where(dist_xr > 0)
    sorted_dist = np.sort(dist_xr.values, axis=1)
    sorted_dist = sorted_dist[:, :k]

    # take the median
    mean_dists = sorted_dist.mean(axis=1)

    return mean_dists


def calculate_median_distance_to_all_cell_types(
        cell_table, dist_xr, k, cell_type_col=settings.CELL_TYPE,
        cell_label_col=settings.CELL_LABEL):
    """Wrapper function to calculate median distance of all cells against all cell types
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

    # call calculate_median_distance_to_cell_type for all cell clusters
    avg_dists = pd.DataFrame(index=cell_table.index.values, columns=all_clusters)
    for cell_cluster in all_clusters:
        avg_dists.loc[:, cell_cluster] = calculate_median_distance_to_cell_type(
            cell_table, dist_xr, cell_cluster, k, cell_type_col, cell_label_col)
        
    return avg_dists


def cell_neighbor_distance_analysis(
        cell_table, dist_mat_dir, save_path, k, fov_col=settings.FOV_ID,
        cell_type_col=settings.CELL_TYPE, cell_label_col=settings.CELL_LABEL):
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
    for fov in fov_list:
        fov_cell_table = cell_table[cell_table[fov_col] == fov]
        fov_dist_xr = xr.load_dataarray(os.path.join(dist_mat_dir, str(fov) + '_dist_mat.xr'))

        # get the average distances between cell types
        fov_cell_dists = calculate_median_distance_to_all_cell_types(
            fov_cell_table, fov_dist_xr, k, cell_type_col, cell_label_col)

        # add the fov name and cell phenotypes to the dataframe
        fov_cell_dists.insert(0, fov_col, fov)
        fov_cell_dists.insert(1, cell_label_col, fov_cell_table[cell_label_col])
        fov_cell_dists.insert(2, cell_type_col, fov_cell_table[cell_type_col])
        cell_dists.append(fov_cell_dists)

    # combine data for all fovs and save to csv
    all_cell_dists = pd.concat(cell_dists)
    all_cell_dists.to_csv(save_path, index=False)

    return all_cell_dists
