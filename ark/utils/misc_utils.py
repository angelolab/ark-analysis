import os
from typing import Any
import warnings
from collections.abc import Iterable
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt


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


def create_invalid_data_str(invalid_data):
    """Creates a easy to read string for ValueError statements.

    Args:
        invalid_data (list[str]): A list of strings containing the invalid / missing data

    Returns:
        str: Returns a formatted string for more detailed ValueError outputs.
    """
    # Holder for the error string
    err_str_data = ""

    # Adding up to 10 invalid values to the err_str_data.
    for idx, data in enumerate(invalid_data[:10], start=1):
        err_msg = "{idx:{fill}{align}{width}} {message}\n".format(
            idx=idx,
            message=data,
            fill=" ",
            align="<",
            width=12,
        )
        err_str_data += err_msg

    return err_str_data


def make_iterable(a: Any, ignore_str: bool = True):
    """ Convert noniterable type to singleton in list

    Args:
        a (T | Iterable[T]):
            value or iterable of type T
        ignore_str (bool):
            whether to ignore the iterability of the str type

    Returns:
        List[T]:
            a as singleton in list, or a if a was already iterable.
    """
    return a if isinstance(a, Iterable) and not ((isinstance(a, str) and ignore_str) or
                                                 isinstance(a, type)) else [a]


def verify_in_list(warn=False, **kwargs):
    """Verify at least whether the values in the first list exist in the second

    Args:
        warn (bool):
            Whether to issue warning instead of error, defaults to False
        **kwargs (list, list):
            Two lists, but will work for single elements as well.
            The first list specified will be tested to see
            if all its elements are contained in the second.

    Raises:
        ValueError:
            if not all values in the first list are found in the second
        Warning:
            if not all values are found and warn is True
    """

    if len(kwargs) != 2:
        raise ValueError("You must provide 2 arguments to verify_in_list")

    test_list, good_values = kwargs.values()
    test_list = list(make_iterable(test_list))
    good_values = list(make_iterable(good_values))

    for v in [test_list, good_values]:
        if len(v) == 0:
            raise ValueError("List arguments cannot be empty")

    if not np.isin(test_list, good_values).all():
        test_list_name, good_values_name = kwargs.keys()
        test_list_name = test_list_name.replace("_", " ")
        good_values_name = good_values_name.replace("_", " ")

        # Calculate the difference between the `test_list` and the `good_values`
        difference = [str(val) for val in test_list if val not in good_values]

        # Only printing up to the first 10 invalid values.
        err_str = ("Not all values given in list {0:^} were found in list {1:^}.\n "
                   "Displaying {2} of {3} invalid value(s) for list {4:^}\n").format(
            test_list_name, good_values_name,
            min(len(difference), 10), len(difference), test_list_name
        )

        err_str += create_invalid_data_str(difference)

        if warn:
            warnings.warn(err_str)
        else:
            raise ValueError(err_str)


def verify_same_elements(enforce_order=False, warn=False, **kwargs):
    """Verify if two lists contain the same elements regardless of count

    Args:
        enforce_order (bool):
            Whether to also check for the same ordering between the two lists
        warn (bool):
            Whether to issue warning instead of error, defaults to False
        **kwargs (list, list):
            Two lists

    Raises:
        ValueError:
            if the two lists don't contain the same elements
    """

    if len(kwargs) != 2:
        raise ValueError("You must provide 2 list arguments to verify_same_elements")

    list_one, list_two = kwargs.values()

    try:
        list_one_cast = list(list_one)
        list_two_cast = list(list_two)
    except TypeError:
        raise ValueError("Both arguments provided must be lists or list types")

    list_one_name, list_two_name = kwargs.keys()
    list_one_name = list_one_name.replace("_", " ")
    list_two_name = list_two_name.replace("_", " ")

    if not np.all(set(list_one_cast) == set(list_two_cast)):
        # Values in list one that are not in list two
        missing_vals_1 = [str(val) for val in (set(list_one_cast) - set(list_two_cast))]

        # Values in list two that are not in list one
        missing_vals_2 = [str(val) for val in (set(list_two_cast) - set(list_one_cast))]

        # Total missing values
        missing_vals_total = [str(val) for val in set(list_one_cast) ^ set(list_two_cast)]

        err_str = (
            "{0} value(s) provided for list {1:^} and list {2:^} are not found in both lists.\n"
        ).format(len(missing_vals_total), list_one_name, list_two_name)

        # Only printing up to the first 10 invalid values for list one.
        err_str += ("{0:>13} \n").format(
            "Displaying {0} of {1} value(s) in list {2} that are missing from list {3}\n".format(
                min(len(missing_vals_1), 10), len(missing_vals_1),
                list_one_name, list_two_name)
        )
        err_str += create_invalid_data_str(missing_vals_1) + "\n"

        # Only printing up to the first 10 invalid values for list two
        err_str += ("{0:>13} \n").format(
            "Displaying {0} of {1} value(s) in list {2} that are missing from list {3}\n".format(
                min(len(missing_vals_2), 10), len(missing_vals_2),
                list_two_name, list_one_name
            )
        )
        err_str += create_invalid_data_str(missing_vals_2) + "\n"

        if warn:
            warnings.warn(err_str)
        else:
            raise ValueError(err_str)
    elif enforce_order and list_one_cast != list_two_cast:
        first_bad_index = next(i for i, (l1, l2) in enumerate(
            zip(list_one_cast, list_two_cast)) if l1 != l2
        )

        err_str = ("Lists %s and %s ordered differently: values %s and %s do not match"
                   " at index %d")

        if warn:
            warnings.warn(err_str % (list_one_name, list_two_name,
                                     list_one_cast[first_bad_index],
                                     list_two_cast[first_bad_index],
                                     first_bad_index))
        else:
            raise ValueError(err_str % (list_one_name, list_two_name,
                                        list_one_cast[first_bad_index],
                                        list_two_cast[first_bad_index],
                                        first_bad_index))
