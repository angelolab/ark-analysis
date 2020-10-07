import os

import numpy as np
import xarray as xr


def combine_xarrays(xarrays, axis):
    """Combines a number of xarrays together

    Args:
        xarrays (tuple):
            a tuple of xarrays
        axis (int):
            either 0, if the xarrays will combined over different fovs, or -1 if they will be
            combined over channels

    Returns:
        xarray.DataArray:
            an xarray that is the combination of all inputs
    """

    first_xr = xarrays[0]
    np_arr = first_xr.values

    # define iterator to hold coord values of dimension that is being stacked
    if axis == 0:
        iterator = first_xr.fovs.values
        shape_slice = slice(1, 4)
    else:
        iterator = first_xr.channels.values
        shape_slice = slice(0, 3)

    # loop through each xarray, stack the coords, and concatenate the values
    for cur_xr in xarrays[1:]:
        cur_arr = cur_xr.values

        if cur_arr.shape[shape_slice] != first_xr.shape[shape_slice]:
            raise ValueError("xarrays have conflicting sizes")

        if axis == 0:
            if not np.array_equal(cur_xr.channels, first_xr.channels):
                raise ValueError("xarrays have different channels")
        else:
            if not np.array_equal(cur_xr.fovs, first_xr.fovs):
                raise ValueError("xarrays have different fovs")

        np_arr = np.concatenate((np_arr, cur_arr), axis=axis)
        if axis == 0:
            iterator = np.append(iterator, cur_xr.fovs.values)
        else:
            iterator = np.append(iterator, cur_xr.channels.values)

    # assign iterator to appropriate coord label
    if axis == 0:
        fovs = iterator
        channels = first_xr.channels.values
    else:
        fovs = first_xr.fovs.values
        channels = iterator

    combined_xr = xr.DataArray(np_arr, coords=[fovs, range(first_xr.shape[1]),
                                               range(first_xr.shape[2]), channels],
                               dims=["fovs", "rows", "cols", "channels"])

    return combined_xr


def combine_fov_directories(dir_path):
    """Combines a folder containing multiple imaging runs into a single folder

    Args:
        dir_path (str):
            path to directory containing the sub directories
    """

    if not os.path.exists(dir_path):
        raise ValueError("Directory does not exist")

    # gets all sub folders
    folders = os.listdir(dir_path)
    folders = [folder for folder in folders if os.path.isdir(os.path.join(dir_path, folder))]

    os.makedirs(os.path.join(dir_path, "combined_folder"))

    # loop through sub folders, get all contents, and transfer to new folder
    for folder in folders:
        fovs = os.listdir(os.path.join(dir_path, folder))
        print(fovs)
        for fov in fovs:
            os.rename(os.path.join(dir_path, folder, fov),
                      os.path.join(dir_path, "combined_folder", folder + "_" + fov))
