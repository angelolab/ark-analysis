import copy
import datetime
from itertools import combinations, product
import json
import os
import random


def set_tiling_params(fov_list_path):
    """Given a file specifying fov regions, set the MIBI tiling parameters

    User inputs will be required for many values

    Args:
        fov_list_path (str):
            Path to the JSON file containing the fov region data

    Returns:
        dict:
            Contains the value associated with each tiling parameter
    """

    # file path validation
    if not os.path.exists(fov_list_path):
        raise FileNotFoundError("FOV region file %s does not exist" % fov_list_path)

    # read in the data
    with open(fov_list_path, 'r') as flf:
        fov_tile_info = json.load(flf)

    # define the parameter dict to return
    tiling_params = {}

    # retrieve the format version
    tiling_params['fovFormatVersion'] = fov_tile_info['fovFormatVersion']

    # define lists to hold the starting x and y coordinates for each fov
    region_start_x = []
    region_start_y = []

    # define lists to hold the number of fovs along each axis
    fov_num_x = []
    fov_num_y = []

    # define lists to hold the size of each fov
    x_fov_size = []
    y_fov_size = []

    # read in the data for each fov (region_start from fov_list_path, fov_num from user)
    for fov in fov_tile_info['fovs']:
        region_start_x.append(fov['centerPointMicrons']['x'])
        region_start_y.append(fov['centerPointMicrons']['y'])

        num_x = int(input("Enter number of x fovs for region %s: " % fov['name']))

        while num_x < 1:
            print("Error: number of fovs must be positive")
            num_x = int(input("Enter number of x fovs for region %s: " % fov['name']))

        num_y = int(input("Enter number of y fovs for region %s: " % fov['name']))

        while num_y < 1:
            print("Error: number of fovs must be positive")
            num_x = int(input("Enter number of y fovs for region %s: " % fov['name']))

        fov_num_x.append(num_x)
        fov_num_y.append(num_y)

        # allow the user to specify the step size
        size_x = int(input("Enter the x step size for region %s: " % fov['name']))

        while size_x < 1:
            print("Error: step size must be positive")
            size_x = int(input("Enter the x step size for region %s: " % fov['name']))

        size_y = int(input("Enter the y step size for region %s: " % fov['name']))

        while size_y < 1:
            print("Error: step size must be positive")
            size_y = int(input("Enter the y step size for region %s: " % fov['name']))

        x_fov_size.append(size_x)
        y_fov_size.append(size_y)

    # allow the user to specify if the FOVs should be randomized
    randomize = int(input("Randomize? Enter 0 for no and 1 for yes: "))

    while randomize not in [0, 1]:
        print("Error: randomize parameter must be 0 or 1")
        randomize = int(input("Randomize? Enter 0 for no and 1 for yes: "))

    # need to copy fov metadata over, needed for create_tiled_regions
    tiling_params['fovs'] = copy.deepcopy(fov_tile_info['fovs'])

    # assign fields to tiling_params
    tiling_params['region_start_x'] = region_start_x
    tiling_params['region_start_y'] = region_start_y
    tiling_params['fov_num_x'] = fov_num_x
    tiling_params['fov_num_y'] = fov_num_y
    tiling_params['x_fov_size'] = x_fov_size
    tiling_params['y_fov_size'] = y_fov_size
    tiling_params['randomize'] = randomize

    # whether to insert moly points between tiles
    # NOTE: moly points will be inserted between different runs regardless of what's set here
    moly_insert = int(
        input("Specify moly point tile interval? Enter 0 for no and 1 for yes: ")
    )

    if moly_insert not in [0, 1]:
        print("Error: moly insertion parameter must enter 0 or 1")
        moly_insert = int(
            input("Specify moly point tile interval? Enter 0 for no and 1 for yes: ")
        )

    # if moly insert is set, we need to specify an additional moly_interval param
    # NOTE: the interval applies regardless of if the tiles overlap runs or not
    if moly_insert:
        moly_interval = int(input("Enter the fov interval size to insert moly points: "))

        while moly_interval < 1:
            print("Error: moly interval must be positive")
            moly_interval = int(input("Enter the fov interval size to insert moly points: "))

        tiling_params['moly_interval'] = moly_interval

    return tiling_params


def create_tiled_regions(tiling_params, moly_path):
    """Create the tiled regions for each fov

    Args:
        tiling_params (dict):
            The tiling parameters created by set_tiling_params
        moly_path (str):
            Path to the JSON moly point file, needed to separate fovs

    Returns:
        dict:
            Data containing information about each tile, will be saved to JSON
    """

    # helper function for creating all pairs between two lists
    def pairs(*lists):
        for t in combinations(lists, 2):
            for pair in product(*t):
                yield pair

    # file path validation
    if not os.path.exists(moly_path):
        raise FileNotFoundError("Moly point file %s does not exist" % moly_path)

    # read in the data
    with open(moly_path, 'r') as mf:
        moly_point = json.load(mf)

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

    # append tile information to fov_list, iterate through each tiles iteratively
    for region_index in range(len(tiling_params['fov_num_x'])):
        # extract start coordinates
        start_x = tiling_params['region_start_x'][region_index]
        start_y = tiling_params['region_start_y'][region_index]

        # generate range of x and y coordinates
        x_range = list(range(tiling_params['fov_num_x'][region_index]))
        y_range = list(range(tiling_params['fov_num_y'][region_index]))

        x_range_rep = x_range * len(y_range)
        y_range_rep = y_range * len(x_range)

        # create all pairs between two lists
        x_y_pairs = list(pairs(x_range, y_range))

        # randomize pairs list if specified
        if tiling_params['randomize'] == 1:
            random.shuffle(x_y_pairs)

        for xi, yi in x_y_pairs:
            # set the current x and y coordinate
            cur_x = start_x + xi * tiling_params['x_fov_size'][region_index]
            cur_y = start_y - yi * tiling_params['y_fov_size'][region_index]

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

        # append moly point to seperate runs if not last
        if region_index != len(tiling_params['fov_num_x']) - 1:
            tiled_regions['fovs'].append(moly_point)

    return tiled_regions
