import os

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns


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


def save_figure(save_dir, save_file, dpi=None):
    """Verify save_dir and save_file, then save to specified location

    Args:
        save_dir (str):
            the name of the directory we wish to save to
        save_file (str):
            the name of the file we wish to save to
        dpi (float):
            the resolution of the figure
    """

    # verify save_dir exists
    if not os.path.exists(save_dir):
        raise FileNotFoundError("save_dir %s does not exist" % save_dir)

    # verify that if save_dir specified, save_file must also be specified
    if save_file is None:
        raise FileNotFoundError("save_dir specified but no save_file specified")

    plt.savefig(os.path.join(save_dir, save_file), dpi=dpi)


def verify_in_list(**kwargs):
    """Verify at least whether the values in the first list exist in the second

    Args:
        **kwargs (list, list):
            Two lists, but will work for single elements as well.
            The first list specified will be tested to see
            if all its elements are contained in the second.```

    Raises:
        ValueError:
            if not all values in the first list are found in the second
    """

    if len(kwargs) != 2:
        raise ValueError("You must provide 2 arguments to verify_in_list")

    test_list, good_values = kwargs.values()

    if not np.isin(test_list, good_values).all():
        bad_vals = ','.join([str(val) for val in test_list if val not in good_values])
        test_list_name, good_values_name = kwargs.keys()
        test_list_name = test_list_name.replace('_', ' ')
        good_values_name = good_values_name.replace('_', ' ')

        err_str = ("Invalid value(s) provided for %s variable: value(s) %s not found"
                   " in %s list")

        raise ValueError(err_str % (test_list_name, bad_vals, good_values_name))


def verify_same_elements(**kwargs):
    """Verify if two lists contain the same elements regardless of count

    Args:
        **kwargs (list, list):
            Two lists

    Raises:
        ValueError:
            if the two lists don't contain the same elements
    """

    if len(kwargs) != 2:
        raise ValueError("You must provide 2 arguments to verify_same_elements")

    list_one, list_two = kwargs.values()

    try:
        list_one_cast = list(list_one)
        list_two_cast = list(list_two)
    except TypeError:
        raise ValueError("Both arguments provided must be lists or list types")

    if not np.all(set(list_one_cast) == set(list_two_cast)):
        bad_vals = ','.join([str(val) for val in set(list_one_cast) ^ set(list_two_cast)])
        list_one_name, list_two_name = kwargs.keys()
        list_one_name = list_one_name.replace('_', ' ')
        list_two_name = list_two_name.replace('_', ' ')

        err_str = ("Invalid value(s) provided in both %s and %s variables: value(s)"
                   " %s not found in both lists")

        raise ValueError(err_str % (list_one_name, list_two_name, bad_vals))
