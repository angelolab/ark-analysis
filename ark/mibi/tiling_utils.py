import copy
import datetime
from itertools import combinations, product
import json
import os
import random


# helper function to reading in input
def read_tiling_param(prompt, error_msg, cond):
    """A helper function to read in tiling input

    Args:
        prompt (str):
            The initial text to display to the user
        error_msg (str):
            The message to display if an invalid input is entered
        cond (function):
            What defines valid input for the variable

    Returns:
        int:
            The value to place in the variable, for now always of type int
    """

    while True:
        var = int(input(prompt))

        if cond(var):
            return var

        print(error_msg)


def generate_region_info(region_start_x, region_start_y, fov_num_x, fov_num_y,
                         x_fov_size, y_fov_size, randomize):
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
        randomize (list):
            Whether to set randomization or not sorted by run

    Returns:
        list:
            The complete set of region_params sorted by run
    """

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
            'randomize': randomize[i]
        }

        # append info to region_params
        region_params_list.append(region_info)

    return region_params_list


def set_tiling_params(fov_list_path, moly_path):
    """Given a file specifying fov regions, set the MIBI tiling parameters

    User inputs will be required for many values. Also returns moly_path data.

    Args:
        fov_list_path (str):
            Path to the JSON file containing the fovs used to define each tiled region
        moly_path (str):
            Path to the JSON moly point file, needed to separate fovs

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

    # read in the data for each fov (region_start from fov_list_path, fov_num from user)
    for fov in fov_tile_info['fovs']:
        region_start_x.append(fov['centerPointMicrons']['x'])
        region_start_y.append(fov['centerPointMicrons']['y'])

        # allow the user to specify the number of fovs along each dimension
        num_x = read_tiling_param(
            "Enter number of x fovs for region %s: " % fov['name'],
            "Error: number of x fovs must be positive",
            lambda nx: nx >= 1
        )

        num_y = read_tiling_param(
            "Enter number of y fovs for region %s: " % fov['name'],
            "Error: number of y fovs must be positive",
            lambda ny: ny >= 1
        )

        fov_num_x.append(num_x)
        fov_num_y.append(num_y)

        # allow the user to specify the step size along each dimension
        size_x = read_tiling_param(
            "Enter the x step size for region %s: " % fov['name'],
            "Error: x step size must be positive",
            lambda sx: sx >= 1
        )

        size_y = read_tiling_param(
            "Enter the y step size for region %s: " % fov['name'],
            "Error: y step size must be positive",
            lambda sy: sy >= 1
        )

        x_fov_size.append(size_x)
        y_fov_size.append(size_y)

        # allow the user to specify if the FOVs should be randomized
        randomize = read_tiling_param(
            "Randomize fovs for region %s? Enter 0 for no and 1 for yes: " % fov['name'],
            "Error: randomize parameter must be 0 or 1",
            lambda r: r in [0, 1]
        )

        region_rand.append(randomize)

    # need to copy fov metadata over, needed for create_tiled_regions
    tiling_params['fovs'] = copy.deepcopy(fov_tile_info['fovs'])

    # store the read in parameters in the region_params key
    tiling_params['region_params'] = generate_region_info(
        region_start_x, region_start_y, fov_num_x, fov_num_y, x_fov_size, y_fov_size, region_rand
    )

    # whether to insert moly points between runs
    moly_run_insert = read_tiling_param(
        "Insert moly points between runs? Enter 0 for no and 1 for yes: ",
        "Error: moly point run parameter must be either 0 or 1",
        lambda mri: mri in [0, 1]
    )

    tiling_params['moly_run'] = moly_run_insert

    # whether to insert moly points between tiles
    moly_interval_insert = read_tiling_param(
        "Specify moly point tile interval? Enter 0 for no and 1 for yes: ",
        "Error: moly interval insertion parameter must enter 0 or 1",
        lambda mii: mii in [0, 1]
    )

    # if moly insert is set, we need to specify an additional moly_interval param
    # NOTE: the interval applies regardless of if the tiles overlap runs or not
    if moly_interval_insert:
        moly_interval = read_tiling_param(
            "Enter the fov interval size to insert moly points: ",
            "Error: moly interval must be positive",
            lambda mi: mi >= 1
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

    all_pairs = []

    for t in combinations((x_range, y_range), 2):
        for pair in product(t[0], t[1]):
            all_pairs.append(pair)

    return all_pairs


def create_tiled_regions(tiling_params, moly_point):
    """Create the tiled regions for each fov

    Args:
        tiling_params (dict):
            The tiling parameters created by set_tiling_params
        moly_point (dict):
            The moly point to insert between fovs (and intervals if specified in tiling_params)

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
        x_range = list(range(region_info['fov_num_x']))
        y_range = list(range(region_info['fov_num_y']))

        x_range_rep = x_range * len(y_range)
        y_range_rep = y_range * len(x_range)

        # create all pairs between two lists
        x_y_pairs = generate_x_y_fov_pairs(x_range, y_range)

        # randomize pairs list if specified
        if region_info['randomize'] == 1:
            random.shuffle(x_y_pairs)

        for xi, yi in x_y_pairs:
            # set the current x and y coordinate
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
        if tiling_params['moly_run'] == 1 and \
           region_index != len(tiling_params['region_params']) - 1:
            tiled_regions['fovs'].append(moly_point)

    return tiled_regions
