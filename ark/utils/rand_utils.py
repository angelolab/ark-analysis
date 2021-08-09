import pandas as pd
import numpy as np
import xarray as xr


def random_sample(data, subset_proportion=0.5, seed=42):
    """Returns a completely random sample of the data

    Args:
        data (pandas.DataArray):
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
