import copy
import datetime
from itertools import combinations, product
import json
import numpy as np
import os
import random

from ark.utils import misc_utils


# helper function to reading in input
def read_tiling_param(prompt, error_msg, cond, dtype):
    """A helper function to read in tiling input

    Args:
        prompt (str):
            The initial text to display to the user
        error_msg (str):
            The message to display if an invalid input is entered
        cond (function):
            What defines valid input for the variable
        dtype (type):
            The type of variable to read

    Returns:
        Union([int, str]):
            The value to place in the variable, limited to just int and str for now
    """

    # ensure the dtype is valid
    misc_utils.verify_in_list(
        provided_dtype=dtype,
        acceptable_dtypes=[int, str]
    )

    while True:
        # read in the variable with correct dtype
        var = dtype(input(prompt))

        # if condition passes, return
        if cond(var):
            return var

        # otherwise, print the error message and re-prompt
        print(error_msg)


def generate_region_info(region_start_x, region_start_y, fov_num_x, fov_num_y,
                         x_fov_size, y_fov_size, region_rand,
                         x_intervals=None, y_intervals=None):
    """Generate the region_params list in the tiling parameter dict

    Args:
        region_start_x (list):
            List of x starting points sorted by run
        region_start_y (list):
            List of y starting points sorted by run
        fov_num_x (list):
            Number of fovs along the x coord sorted by run
        fov_num_y (list):
            Number of fovs along the y coord sorted by run
        x_fov_size (list):
            The size of the x axis sorted by run
        y_fov_size (list):
            The size of the y axis sorted by run
        region_rand (list):
            Whether to set randomization or not sorted by run
        x_intervals (list):
            List of x points to choose sorted by run
        y_intervals (list):
            List of y points to choose sorted by run

    Returns:
        list:
            The complete set of region_params sorted by run
    """

    # check that region_end_x and region_end_y are either both None or set
    # cannot use ^ shorthand because we're not dealing with bools
    if (x_intervals and not y_intervals) or (not x_intervals and y_intervals):
        raise ValueError("x_intervals and y_intervals must either both be None or set")

    region_params_list = []

    # iterate over all the region parameters, all parameter lists are the same length
    for i in range(len(region_start_x)):
        # define a dict containing all the region info for the specific fov
        region_info = {
            'region_start_x': region_start_x[i],
            'region_start_y': region_start_y[i],
            'fov_num_x': fov_num_x[i],
            'fov_num_y': fov_num_y[i],
            'x_fov_size': x_fov_size[i],
            'y_fov_size': y_fov_size[i],
            'region_rand': region_rand[i]
        }

        # add x_interval and y_interval if they're set
        if x_intervals and y_intervals:
            region_info['x_interval'] = x_intervals[i]
            region_info['y_interval'] = y_intervals[i]

        # append info to region_params
        region_params_list.append(region_info)

    return region_params_list


def read_tma_region_input(fov_tile_info, region_start_x, region_start_y,
                          region_end_x, region_end_y, fov_num_x, fov_num_y, x_fov_size,
                          y_fov_size, x_intervals, y_intervals, region_rand):
    """Reads input for TMAs from user and fov_tile_info

    Updates all the tiling params inplace

    Args:
        fov_tile_info (dict):
            The data containing the fovs used to define each tiled region
        region_start_x (list):
            List of x starting points sorted by run
        region_start_y (list):
            List of y starting points sorted by run
        region_end_x (list):
            List of x ending points sorted by run
        region_end_y (list):
            List of y ending points sorted by run
        fov_num_x (list):
            Number of fovs along the x coord sorted by run
        fov_num_y (list):
            Number of fovs along the y coord sorted by run
        x_fov_size (list):
            The size of the x axis sorted by run
        y_fov_size (list):
            The size of the y axis sorted by run
        x_intervals (list):
            The indices to put fovs at sorted by run
        y_intervals (list):
            The indices to put fovs at sorted by run
        region_rand (list):
            Whether to set randomization or not sorted by run
    """

    # there has to be a starting and ending fov for each region
    if len(fov_tile_info['fovs']) % 2 != 0:
        raise ValueError(
            "Data in fov_list_path needs to contain a start and end fov for each region"
        )

    # every two fovs should define the start and end of the fov
    for i in range(0, len(fov_tile_info['fovs']), 2):
        # define the current start and end fov
        fov_batches = fov_tile_info['fovs'][i:i + 2]
        start_fov = fov_batches[0]
        end_fov = fov_batches[1]

        # define the start and end coordinates
        start_fov_x = start_fov['centerPointMicrons']['x']
        end_fov_x = end_fov['centerPointMicrons']['x']
        start_fov_y = start_fov['centerPointMicrons']['y']
        end_fov_y = end_fov['centerPointMicrons']['y']

        # the coordinates have to be valid
        if start_fov_x > end_fov_x or start_fov_y > end_fov_y:
            err_msg = ("Coordinate error for region %s: start coordinates cannot be",
                       " greater than end coordinates")
            raise ValueError(err_msg % start_fov['name'])

        region_start_x.append(start_fov_x)
        region_start_y.append(start_fov_y)
        region_end_x.append(end_fov_x)
        region_end_y.append(end_fov_y)

        # the num_x, num_y, size_x, and size_y need additional validation
        # since they may not be compatible with the starting and ending coordinates
        while True:
            # allow the user to specify the number of fovs along each dimension
            num_x = read_tiling_param(
                "Enter number of x fovs for region %s (at least 3 required): " % start_fov['name'],
                "Error: number of x fovs must be 3 or more",
                lambda nx: nx >= 3,
                dtype=int
            )

            num_y = read_tiling_param(
                "Enter number of y fovs for region %s (at least 3 required): " % start_fov['name'],
                "Error: number of y fovs must be 3 or more",
                lambda ny: ny >= 3,
                dtype=int
            )

            # allow the user to specify the step size along each dimension
            size_x = read_tiling_param(
                "Enter the x image size for region %s: " % start_fov['name'],
                "Error: x step size must be positive",
                lambda sx: sx >= 1,
                dtype=int
            )

            size_y = read_tiling_param(
                "Enter the y image size for region %s: " % start_fov['name'],
                "Error: y step size must be positive",
                lambda sy: sy >= 1,
                dtype=int
            )

            # find num_x/num_y even intervals between start and end fov_x/fov_y
            # casted because indices cannot be floats
            x_interval = np.linspace(start_fov_x, end_fov_x, num_x).astype(int)
            y_interval = np.linspace(start_fov_y, end_fov_y, num_y).astype(int)

            # get difference between x and y
            x_spacing = x_interval[1] - x_interval[0]
            y_spacing = y_interval[1] - y_interval[0]

            # we're good to go if size_x is not greater than x_spacing and y_spacing
            if size_x <= x_spacing and size_y <= y_spacing:
                break

            # otherwise throw errors for invalid num_x/num_y and size_x/size_y
            if size_x > x_spacing:
                err_msg = ("Provided params num_x = %d, size_x = %d are incompatible"
                           " with x start = %d and x end = %d for region %s")
                print(err_msg % (num_x, size_x, start_fov_x, end_fov_x, start_fov['name']))

            if size_y > y_spacing:
                err_msg = ("Provided params num_y = %d, size_y = %d are incompatible"
                           " with y start = %d and y end = %d for region %s")
                print(err_msg % (num_y, size_y, start_fov_y, end_fov_y, start_fov['name']))

        fov_num_x.append(num_x)
        fov_num_y.append(num_y)

        x_fov_size.append(size_x)
        y_fov_size.append(size_y)

        x_intervals.append(list(x_interval))
        y_intervals.append(list(y_interval))

        # allow the user to specify if the FOVs should be randomized
        randomize = read_tiling_param(
            "Randomize fovs for region %s? Y/N: " % start_fov['name'],
            "Error: randomize parameter must Y or N",
            lambda r: r in ['Y', 'N'],
            dtype=str
        )

        region_rand.append(randomize)


def read_non_tma_region_input(fov_tile_info, region_start_x, region_start_y,
                              fov_num_x, fov_num_y, x_fov_size, y_fov_size, region_rand):
    """Reads input for non-TMAs from user and fov_tile_info

    Updates all the tiling params inplace

    Args:
        fov_tile_info (dict):
            The data containing the fovs used to define each tiled region
        region_start_x (list):
            List of x starting points sorted by run
        region_start_y (list):
            List of y starting points sorted by run
        fov_num_x (list):
            Number of fovs along the x coord sorted by run
        fov_num_y (list):
            Number of fovs along the y coord sorted by run
        x_fov_size (list):
            The size of the x axis sorted by run
        y_fov_size (list):
            The size of the y axis sorted by run
        region_rand (list):
            Whether to set randomization or not sorted by run
    """

    # read in the data for each fov (region_start from fov_list_path, fov_num from user)
    for fov in fov_tile_info['fovs']:
        region_start_x.append(fov['centerPointMicrons']['x'])
        region_start_y.append(fov['centerPointMicrons']['y'])

        # allow the user to specify the number of fovs along each dimension
        num_x = read_tiling_param(
            "Enter number of x fovs for region %s: " % fov['name'],
            "Error: number of x fovs must be positive",
            lambda nx: nx >= 1,
            dtype=int
        )

        num_y = read_tiling_param(
            "Enter number of y fovs for region %s: " % fov['name'],
            "Error: number of y fovs must be positive",
            lambda ny: ny >= 1,
            dtype=int
        )

        fov_num_x.append(num_x)
        fov_num_y.append(num_y)

        # allow the user to specify the step size along each dimension
        size_x = read_tiling_param(
            "Enter the x step size for region %s: " % fov['name'],
            "Error: x step size must be positive",
            lambda sx: sx >= 1,
            dtype=int
        )

        size_y = read_tiling_param(
            "Enter the y step size for region %s: " % fov['name'],
            "Error: y step size must be positive",
            lambda sy: sy >= 1,
            dtype=int
        )

        x_fov_size.append(size_x)
        y_fov_size.append(size_y)

        # allow the user to specify if the FOVs should be randomized
        randomize = read_tiling_param(
            "Randomize fovs for region %s? Y/N: " % fov['name'],
            "Error: randomize parameter must Y or N",
            lambda r: r in ['Y', 'N'],
            dtype=str
        )

        region_rand.append(randomize)


def set_tiling_params(fov_list_path, moly_path, tma=False):
    """Given a file specifying fov regions, set the MIBI tiling parameters

    User inputs will be required for many values. Also returns moly_path data.

    Args:
        fov_list_path (str):
            Path to the JSON file containing the fovs used to define each tiled region
        moly_path (str):
            Path to the JSON moly point file, needed to separate fovs
        tma (bool):
            Whether the data in fov_list_path is in TMA format or not

    Returns:
        tuple:
            Contains:

            - A dict containing the tiling parameters for each fov
            - A dict defining the moly points to insert if specified
    """

    # file path validation
    if not os.path.exists(fov_list_path):
        raise FileNotFoundError("FOV region file %s does not exist" % fov_list_path)

    if not os.path.exists(moly_path):
        raise FileNotFoundError("Moly point file %s does not exist" % moly_path)

    # read in the fov list data
    with open(fov_list_path, 'r') as flf:
        fov_tile_info = json.load(flf)

    # read in the moly point data
    with open(moly_path, 'r') as mpf:
        moly_point = json.load(mpf)

    # define the parameter dict to return
    tiling_params = {}

    # retrieve the format version
    tiling_params['fovFormatVersion'] = fov_tile_info['fovFormatVersion']

    # define lists to hold the starting x and y coordinates for each region
    region_start_x = []
    region_start_y = []

    # define lists to hold the number of fovs along each axis
    fov_num_x = []
    fov_num_y = []

    # define lists to hold the size of each fov
    x_fov_size = []
    y_fov_size = []

    # define a list to determine if the fovs should be randomly ordered
    region_rand = []

    # if TMA is set we also need to define ending coordinates and intervals for each region
    region_end_x = [] if tma else None
    region_end_y = [] if tma else None
    x_intervals = [] if tma else None
    y_intervals = [] if tma else None

    # reading in the params from fov_tile_info will be different for TMAs and non-TMAs
    if tma:
        read_tma_region_input(
            fov_tile_info, region_start_x, region_start_y, region_end_x, region_end_y,
            fov_num_x, fov_num_y, x_fov_size, y_fov_size, x_intervals, y_intervals, region_rand
        )
    else:
        read_non_tma_region_input(
            fov_tile_info, region_start_x, region_start_y, fov_num_x, fov_num_y,
            x_fov_size, y_fov_size, region_rand
        )

    # need to copy fov metadata over, needed for create_tiled_regions
    tiling_params['fovs'] = copy.deepcopy(fov_tile_info['fovs'])

    # store the read in parameters in the region_params key
    tiling_params['region_params'] = generate_region_info(
        region_start_x, region_start_y, fov_num_x, fov_num_y, x_fov_size, y_fov_size, region_rand,
        x_intervals, y_intervals
    )

    # whether to insert moly points between runs
    moly_run_insert = read_tiling_param(
        "Insert moly points between runs? Y/N: ",
        "Error: moly point run parameter must be either Y or N",
        lambda mri: mri in ['Y', 'N'],
        dtype=str
    )

    tiling_params['moly_run'] = moly_run_insert

    # whether to insert moly points between tiles
    moly_interval_insert = read_tiling_param(
        "Specify moly point tile interval? Y/N: ",
        "Error: moly interval insertion parameter must either Y or N",
        lambda mii: mii in ['Y', 'N'],
        dtype=str
    )

    # if moly insert is set, we need to specify an additional moly_interval param
    # NOTE: the interval applies regardless of if the tiles overlap runs or not
    if moly_interval_insert == 'Y':
        moly_interval = read_tiling_param(
            "Enter the fov interval size to insert moly points: ",
            "Error: moly interval must be positive",
            lambda mi: mi >= 1,
            dtype=int
        )

        tiling_params['moly_interval'] = moly_interval

    return tiling_params, moly_point


# helper function for creating all pairs between two lists
def generate_x_y_fov_pairs(x_range, y_range):
    """Given all x and y coordinates a fov can take, generate all possible (x, y) pairings

    Args:
        x_range (list):
            Range of x values a fov can take
        y_range (list):
            Range of y values a fov can take

    Returns:
        list:
            Every possible (x, y) pair for a fov
    """

    # define a list to hold all the (x, y) pairs
    all_pairs = []

    # iterate over all combinations of x and y
    for t in combinations((x_range, y_range), 2):
        # compute the product of the resulting x and y list pair, append results
        for pair in product(t[0], t[1]):
            all_pairs.append(pair)

    return all_pairs


def create_tiled_regions(tiling_params, moly_point, tma=False):
    """Create the tiled regions for each fov for non-TMAs

    Args:
        tiling_params (dict):
            The tiling parameters created by set_tiling_params
        moly_point (dict):
            The moly point to insert between fovs (and intervals if specified in tiling_params)
        tma (bool):
            Whether the data in tiling_params is in TMA format or not

    Returns:
        dict:
            Data containing information about each tile, will be saved to JSON
    """

    # get the current time info
    dt = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    # define the fov tiling info
    tiled_regions = {
        'exportDateTime': dt,
        'fovFormatVersion': tiling_params['fovFormatVersion'],
        'fovs': []
    }

    # define a counter to determine where to insert a moly point
    # only used if tiling_params['moly_interval'] is set
    moly_counter = 0

    # iterate through each region and append created tiles to tiled_regions['fovs']
    for region_index, region_info in enumerate(tiling_params['region_params']):
        # extract start coordinates
        start_x = region_info['region_start_x']
        start_y = region_info['region_start_y']

        # generate range of x and y coordinates
        if tma:
            x_range = region_info['x_interval']
            y_range = region_info['y_interval']
        else:
            x_range = list(range(region_info['fov_num_x']))
            y_range = list(range(region_info['fov_num_y']))

        # create all pairs between two lists
        x_y_pairs = generate_x_y_fov_pairs(x_range, y_range)

        # randomize pairs list if specified
        if region_info['region_rand'] == 'Y':
            random.shuffle(x_y_pairs)

        for xi, yi in x_y_pairs:
            # set the current x and y coordinate
            if tma:
                cur_x = xi.item()
                cur_y = yi.item()
            else:
                cur_x = start_x + xi * region_info['x_fov_size']
                cur_y = start_y + yi * region_info['y_fov_size']

            # copy the fov metadata over and add cur_x, cur_y, and identifier
            fov = copy.deepcopy(tiling_params['fovs'][region_index])
            fov['centerPointMicrons']['x'] = cur_x
            fov['centerPointMicrons']['y'] = cur_y
            fov['name'] = f'row{yi}_col{xi}'

            # append value to tiled_regions
            tiled_regions['fovs'].append(fov)

            # increment moly_counter as we've added another fov
            moly_counter += 1

            # append a moly point if moly_interval is set and we've reached the interval threshold
            if 'moly_interval' in tiling_params and \
               moly_counter % tiling_params['moly_interval'] == 0:
                tiled_regions['fovs'].append(moly_point)

        # append moly point to seperate runs if not last and if specified
        if tiling_params['moly_run'] == 'Y' and \
           region_index != len(tiling_params['region_params']) - 1:
            tiled_regions['fovs'].append(moly_point)

    return tiled_regions
