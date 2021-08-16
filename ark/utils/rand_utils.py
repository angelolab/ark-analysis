import numpy as np
import pandas as pd
import xarray as xr


def random_sample(data, subset_proportion=0.5, seed=42):
    """Returns a completely random sample of the data

    Args:
        data (pandas.DataFrame):
            The data to randomly sample
        subset_proportion (float):
            The percentage of rows to take from the data
        seed (float):
            The randomization factor

    Returns:
        pandas.DataArray:
            A random subset of data
    """

    return data.sample(frac=subset_proportion, random_state=seed)


def stratified_random_sample(data, groupby, subset_proportion=0.5, seed=42):
    """Returns a random sample stratified by the proportion of values in a column

    Args:
        data (pandas.DataArray):
            The data to randomly sample
        groupby (str):
            The name of the column to stratify over
        subset_proportion (float):
            The percentage of rows to take from the data
        seed (float):
            The randomization factor

    Returns:
        pandas.DataArray:
            A random stratified subset of data
    """
    pass


def pixel_cluster_imbalanced_sample(fov_disease_info, preprocessed_path, subsetted_path):
    pass


def spatial_lda_imbalanced_sample(fov_disease_info):
    pass
