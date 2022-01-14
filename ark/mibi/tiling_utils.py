import copy
from IPython.display import display
import ipywidgets as widgets
from itertools import combinations, product
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.draw import ellipse
from sklearn.utils import shuffle

import ark.settings as settings
from ark.utils import misc_utils


def assign_metadata_vals(input_dict, output_dict, keys_ignore):
    """Copy the `str`, `int`, `float`, and `bool` metadata keys of
    `input_dict` over to `output_dict`

    Args:
        input_dict (dict):
            The `dict` to copy the metadata values over from
        output_dict (dict):
            The `dict` to copy the metadata values into.
            Note that if a metadata key name in `input_dict` exists in `output_dict`,
            the latter's will get overwritten
        keys_ignore (list):
            The list of keys in input_dict to ignore

    Returns:
        dict:
            `output_dict` with the valid `metadata_keys` from `input_dict` copied over
    """

    # get the metadata values to copy over
    metadata_keys = list(input_dict.keys())

    # remove anything set in keys_ignore
    for ki in keys_ignore:
        if ki in input_dict:
            metadata_keys.remove(ki)

    # assign over the remaining metadata keys
    for mk in metadata_keys:
        if type(input_dict[mk]) in [str, int, float, bool, type(None)]:
            output_dict[mk] = input_dict[mk]

    return output_dict


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
        # print error message and re-prompt if cannot be coerced
        try:
            var = dtype(input(prompt))
        except ValueError:
            print(error_msg)
            continue

        # if condition passes, return
        if cond(var):
            return var

        # otherwise, print the error message and re-prompt
        print(error_msg)


def generate_region_info(region_params):
    """Generate the `region_params` list in the tiling parameter dict

    Args:
        region_params (dict):
            A `dict` mapping each region-specific parameter to a list of values per FOV

    Returns:
        list:
            The complete set of `region_params` sorted by region
    """

    # define the region params list
    region_params_list = []

    # iterate over all the region parameters, all parameter lists are the same length
    for i in range(len(region_params['region_start_x'])):
        # define a dict containing all the region info for the specific FOV
        region_info = {
            rp: region_params[rp][i] for rp in region_params
        }

        # append info to region_params
        region_params_list.append(region_info)

    return region_params_list


def tiled_region_read_input(fov_list_info, region_params):
    """Reads input for tiled regions from user and `fov_list_info`.

    Updates all the tiling params inplace. Units used are microns.

    Args:
        fov_list_info (dict):
            The data containing the FOVs used to define each tiled region
        region_params (dict):
            A `dict` mapping each region-specific parameter to a list of values per FOV
    """

    # read in the data for each fov (region_start from fov_list_path, all others from user)
    for fov in fov_list_info['fovs']:
        region_params['region_start_x'].append(fov['centerPointMicrons']['x'])
        region_params['region_start_y'].append(fov['centerPointMicrons']['y'])

        # allow the user to specify the number of fovs along each dimension
        num_x = read_tiling_param(
            "Enter number of x FOVs for region %s: " % fov['name'],
            "Error: number of x FOVs must be a positive integer",
            lambda nx: nx >= 1,
            dtype=int
        )

        num_y = read_tiling_param(
            "Enter number of y FOVs for region %s: " % fov['name'],
            "Error: number of y FOVs must be a positive integer",
            lambda ny: ny >= 1,
            dtype=int
        )

        region_params['fov_num_x'].append(num_x)
        region_params['fov_num_y'].append(num_y)

        # allow the user to specify the step size along each dimension
        size_x = read_tiling_param(
            "Enter the x step size for region %s (in microns): " % fov['name'],
            "Error: x step size must be a positive integer",
            lambda sx: sx >= 1,
            dtype=int
        )

        size_y = read_tiling_param(
            "Enter the y step size for region %s (in microns): " % fov['name'],
            "Error: y step size must be a positive integer",
            lambda sy: sy >= 1,
            dtype=int
        )

        region_params['x_fov_size'].append(size_x)
        region_params['y_fov_size'].append(size_y)

        # allow the user to specify if the FOVs should be randomized
        randomize = read_tiling_param(
            "Randomize FOVs for region %s? Y/N: " % fov['name'],
            "Error: randomize parameter must Y or N",
            lambda r: r in ['Y', 'N', 'y', 'n'],
            dtype=str
        )

        randomize = randomize.upper()

        region_params['region_rand'].append(randomize)


def tiled_region_set_params(fov_list_path, moly_path):
    """Given a file specifying FOV regions, set the MIBI tiling parameters

    User inputs will be required for many values. Also returns `moly_path` data.

    Args:
        fov_list_path (str):
            Path to the JSON file containing the FOVs used to define each tiled region
        moly_path (str):
            Path to the JSON moly point file, needed to separate FOVs

    Returns:
        tuple:
            Contains:

            - A `dict` containing the tiling parameters for each FOV
            - A `dict` defining the moly points to insert if specified
    """

    # file path validation
    if not os.path.exists(fov_list_path):
        raise FileNotFoundError("FOV region file %s does not exist" % fov_list_path)

    if not os.path.exists(moly_path):
        raise FileNotFoundError("Moly point file %s does not exist" % moly_path)

    # read in the fov list data
    with open(fov_list_path, 'r') as flf:
        fov_list_info = json.load(flf)

    # read in the moly point data
    with open(moly_path, 'r') as mpf:
        moly_point = json.load(mpf)

    # define the parameter dict to return
    tiling_params = {}

    # copy over the metadata values from fov_list_info to tiling_params
    tiling_params = assign_metadata_vals(fov_list_info, tiling_params, ['fovs'])

    # define the region_params dict
    region_params = {rpf: [] for rpf in settings.REGION_PARAM_FIELDS}

    # prompt the user for params associated with each tiled region
    tiled_region_read_input(fov_list_info, region_params)

    # need to copy fov metadata over, needed for generate_fov_list
    tiling_params['fovs'] = copy.deepcopy(fov_list_info['fovs'])

    # store the read in parameters in the region_params key
    tiling_params['region_params'] = generate_region_info(region_params)

    # whether to insert moly points between regions
    moly_region_insert = read_tiling_param(
        "Insert moly points between regions? Y/N: ",
        "Error: moly point region parameter must be either Y or N",
        lambda mri: mri in ['Y', 'N', 'y', 'n'],
        dtype=str
    )

    # convert to uppercase to standardize
    moly_region_insert = moly_region_insert.upper()
    tiling_params['moly_region'] = moly_region_insert

    # whether to insert moly points between fovs
    moly_interval_insert = read_tiling_param(
        "Specify moly point FOV interval? Y/N: ",
        "Error: moly interval insertion parameter must either Y or N",
        lambda mii: mii in ['Y', 'N', 'y', 'n'],
        dtype=str
    )

    # convert to uppercase to standardize
    moly_interval_insert = moly_interval_insert.upper()

    # if moly insert is set, we need to specify an additional moly_interval param
    # NOTE: the interval applies regardless of if the fovs overlap regions or not
    if moly_interval_insert == 'Y':
        moly_interval = read_tiling_param(
            "Enter the FOV interval size to insert moly points: ",
            "Error: moly interval must be a positive integer",
            lambda mi: mi >= 1,
            dtype=int
        )

        tiling_params['moly_interval'] = moly_interval

    return tiling_params, moly_point


def generate_x_y_fov_pairs(x_range, y_range):
    """Given all x and y coordinates a FOV can take, generate all possible `(x, y)` pairings

    Args:
        x_range (list):
            Range of x values a FOV can take
        y_range (list):
            Range of y values a FOV can take

    Returns:
        list:
            Every possible `(x, y)` pair for a FOV
    """

    # define a list to hold all the (x, y) pairs
    all_pairs = []

    # iterate over all combinations of x and y
    for t in combinations((x_range, y_range), 2):
        # compute the product of the resulting x and y list pair, append results
        for pair in product(t[0], t[1]):
            all_pairs.append(pair)

    return all_pairs


def tiled_region_generate_fov_list(tiling_params, moly_point):
    """Generate the list of FOVs on the image from the `tiling_params` set for tiled regions

    Moly point insertion: happens once every number of FOVs you specified in
    `tiled_region_set_params`. There are a couple caveats to keep in mind:

    - The interval specified will not reset between regions. In other words, if the interval is 3
      and the next set of FOVs contains 2 in region 1 and 1 in region 2, the next Moly point will
      be placed after the 1st FOV in region 2 (not after the 3rd FOV in region 2). Moly points
      inserted between regions are ignored in this calculation.
    - If the interval specified cleanly divides the number of FOVs in a region, a Moly point will
      not be placed at the end of the region. Suppose 3 FOVs are defined along both the x- and
      y-axis for region 1 (for a total of 9 FOVs) and a Moly point FOV interval of 3 is specified.
      Without also setting Moly point insertion between different regions, a Moly point will NOT be
      placed after the last FOV of region 1 (the next Moly point will appear in region 2's FOVs).

    Args:
        tiling_params (dict):
            The tiling parameters created by `tiled_region_set_params`
        moly_point (dict):
            The moly point to insert between FOVs (and intervals if specified in `tiling_params`)

    Returns:
        dict:
            Data containing information about each FOV
    """

    # define the fov_regions dict
    fov_regions = {}

    # copy over the metadata values from tiling_params to fov_regions
    fov_regions = assign_metadata_vals(
        tiling_params, fov_regions, ['region_params', 'moly_region', 'moly_interval']
    )

    # define a specific FOVs field in fov_regions, this will contain the actual FOVs
    fov_regions['fovs'] = []

    # define a counter to determine where to insert a moly point
    # only used if moly_interval is set in tiling_params
    # NOTE: total_fovs is used to prevent moly_counter from initiating the addition of
    # a Moly point at the end of a region
    moly_counter = 0
    total_fovs = 0

    # iterate through each region and append created fovs to fov_regions['fovs']
    for region_index, region_info in enumerate(tiling_params['region_params']):
        # extract start coordinates
        start_x = region_info['region_start_x']
        start_y = region_info['region_start_y']

        # define the range of x- and y-coordinates to use
        x_range = list(range(region_info['fov_num_x']))
        y_range = list(range(region_info['fov_num_y']))

        # create all pairs between two lists
        x_y_pairs = generate_x_y_fov_pairs(x_range, y_range)

        # name the FOVs according to MIBI conventions
        fov_names = ['R%dC%d' % (y + 1, x + 1) for x in range(region_info['fov_num_x'])
                     for y in range(region_info['fov_num_y'])]

        # randomize pairs list if specified
        if region_info['region_rand'] == 'Y':
            # make sure the fov_names are set in the same shuffled indices for renaming
            x_y_pairs, fov_names = shuffle(x_y_pairs, fov_names)

        # update total_fovs, we'll prevent moly_counter from triggering the appending of
        # a Moly point at the end of a region this way
        total_fovs += len(x_y_pairs)

        for index, (xi, yi) in enumerate(x_y_pairs):
            # use the fov size to scale to the current x- and y-coordinate
            cur_x = start_x + xi * region_info['x_fov_size']
            cur_y = start_y - yi * region_info['y_fov_size']

            # copy the fov metadata over and add cur_x, cur_y, and name
            fov = copy.deepcopy(tiling_params['fovs'][region_index])
            fov['centerPointMicrons']['x'] = cur_x
            fov['centerPointMicrons']['y'] = cur_y
            fov['name'] = fov_names[index]

            # append value to fov_regions
            fov_regions['fovs'].append(fov)

            # increment moly_counter as we've added another fov
            moly_counter += 1

            # append a Moly point if moly_interval is set and we've reached the interval threshold
            # the exception: don't insert a Moly point at the end of a region
            if 'moly_interval' in tiling_params and \
               moly_counter % tiling_params['moly_interval'] == 0 and \
               moly_counter < total_fovs:
                fov_regions['fovs'].append(moly_point)

        # append Moly point to seperate regions if not last and if specified
        if tiling_params['moly_region'] == 'Y' and \
           region_index != len(tiling_params['region_params']) - 1:
            fov_regions['fovs'].append(moly_point)

    return fov_regions


def tma_generate_fov_list(fov_list_path, num_fov_x, num_fov_y):
    """Generate the list of FOVs on the image using the TMA input file in `fov_list_path`

    NOTE: unlike tiled regions, the returned list of FOVs is just an intermediate step to the
    interactive remapping process. So the result will just be each FOV name mapped to its centroid.

    Args:
        fov_list_path (dict):
            Path to the JSON file containing the FOVs used to define the tiled TMA region
        num_fov_x (int):
            Number of FOVs to define along the x-axis
        num_fov_y (int):
            Number of FOVs to define along the y-axis

    Returns:
        dict:
            Data containing information about each FOV (just FOV name mapped to centroid)
    """

    # file path validation
    if not os.path.exists(fov_list_path):
        raise FileNotFoundError("FOV region file %s does not exist" % fov_list_path)

    # user needs to define at least 3 FOVs along the x- and y-axes
    if num_fov_x < 3:
        raise ValueError("Number of FOVs along x-axis must be at least 3")

    if num_fov_y < 3:
        raise ValueError("Number of FOVs along y-axis must be at least 3")

    # read in fov_list_path
    with open(fov_list_path, 'r') as flf:
        fov_list_info = json.load(flf)

    # a TMA can only be defined by 2 FOVs: an upper-left corner and a bottom-right corner
    if len(fov_list_info['fovs']) != 2:
        raise ValueError("Your FOV region file %s needs to contain only 2 FOVs" % fov_list_path)

    # retrieve the corner FOVs
    # NOTE: the upper-left should always be listed before the bottom-right
    upper_left = fov_list_info['fovs'][0]
    bottom_right = fov_list_info['fovs'][1]

    # define the start and end coordinates
    start_fov_x = upper_left['centerPointMicrons']['x']
    end_fov_x = bottom_right['centerPointMicrons']['x']
    start_fov_y = upper_left['centerPointMicrons']['y']
    end_fov_y = bottom_right['centerPointMicrons']['y']

    # the coordinates have to be valid: upper-left cannot be below or to the right of bottom-right
    if start_fov_x > end_fov_x:
        err_msg = ("Coordinate error for region %s: upper-left x coordinates cannot be"
                   " greater than bottom-right coordinates")
        raise ValueError(err_msg % upper_left['name'])

    # NOTE: because ascending values on the y-axis go from bottom to top
    # we need to enforce a < rather than > constraint
    if start_fov_y < end_fov_y:
        err_msg = ("Coordinate error for region %s: upper-left y coordinates cannot be"
                   " less than bottom-right coordinates")
        raise ValueError(err_msg % upper_left['name'])

    # define each FOV along the x- and y-axis, casted because indices cannot be floats
    # need additional .item() cast to prevent int64 is not JSON serializable error
    x_interval = [
        x.item() for x in np.linspace(start_fov_x, end_fov_x, num_fov_x).astype(int)
    ]
    y_interval = list(reversed([
        y.item() for y in np.linspace(end_fov_y, start_fov_y, num_fov_y).astype(int)
    ]))

    # create all pairs between two lists
    x_y_pairs = generate_x_y_fov_pairs(x_interval, y_interval)

    # name the FOVs according to MIBI conventions
    fov_names = ['R%dC%d' % (y + 1, x + 1) for x in range(num_fov_x) for y in range(num_fov_y)]

    # define the fov_regions dict
    fov_regions = {}

    # map each name to its corresponding coordinate value
    for index, (xi, yi) in enumerate(x_y_pairs):
        fov_regions[fov_names[index]] = (xi, yi)

    return fov_regions


def convert_microns_to_pixels(coord):
    """Convert the coordinate in stage microns to optical pixels.

    In other words, co-register using the centroid of a FOV.

    The values are coerced to ints to allow indexing into the slide.
    Coordinates are also returned in `(y, x)` form to account for a different coordinate axis.

    Args:
        coord (tuple):
            The coordinate in microns to convert

    Returns:
        tuple:
            The converted coordinate from microns to pixels
    """

    # NOTE: all conversions are done using the fiducials
    # convert from microns to stage coordinates
    stage_coord_x = (
        coord[0] * settings.MICRON_TO_STAGE_X_MULTIPLIER - settings.MICRON_TO_STAGE_X_OFFSET
    )
    stage_coord_y = (
        coord[1] * settings.MICRON_TO_STAGE_Y_MULTIPLIER - settings.MICRON_TO_STAGE_Y_OFFSET
    )

    # convert from stage coordinates to pixels
    pixel_coord_x = (
        stage_coord_x + settings.STAGE_TO_PIXEL_X_OFFSET
    ) * settings.STAGE_TO_PIXEL_X_MULTIPLIER
    pixel_coord_y = (
        stage_coord_y + settings.STAGE_TO_PIXEL_Y_OFFSET
    ) * settings.STAGE_TO_PIXEL_Y_MULTIPLIER

    return (int(pixel_coord_y), int(pixel_coord_x))


def assign_closest_fovs(manual_fovs, auto_fovs):
    """For each FOV in `manual_fovs`, map it to its closest FOV in `auto_fovs`

    Args:
        manual_fovs (dict):
            The list of FOVs proposed by the user
        auto_fovs (dict):
            The list of FOVs generated by `set_tiling_params` run in
            `example_fov_grid_generate.ipynb`

    Returns:
        tuple:

        - A `dict` defining the mapping of FOV names between `manual_fovs` and `auto_fovs`
        - A `dict` defining each FOV in `manual_fovs` mapped to its centroid coordinates
        - A `dict` defining each FOV in `auto_fovs` mapped to its centroid coordinates
    """

    # define the converted centroid info for manual_fovs and auto_fovs
    # NOTE: manual FOVs is defined as a normal run file would, but auto FOVs
    # is defined as simply the FOV name to its centroid coordinate
    manual_fovs_info = {
        fov['name']: convert_microns_to_pixels(
            (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
        )

        for fov in manual_fovs['fovs']
    }

    auto_fovs_info = {
        fov: convert_microns_to_pixels(
            (auto_fovs[fov][0], auto_fovs[fov][1])
        )

        for fov in auto_fovs
    }

    # we define these "reverse" dicts to map from centroid back to fov name
    # this makes it easier to use numpy broadcasting to help find the closest fov pairs
    manual_centroid_to_fov = {
        (manual_fovs_info[fov][0], manual_fovs_info[fov][1]): fov
        for fov in manual_fovs_info
    }

    auto_centroid_to_fov = {
        (auto_fovs_info[fov][0], auto_fovs_info[fov][1]): fov
        for fov in auto_fovs_info
    }

    # define numpy arrays of the manual and auto centroids
    manual_centroids = np.array(
        [list(centroid) for centroid in manual_centroid_to_fov]
    )
    auto_centroids = np.array(
        [list(centroid) for centroid in auto_centroid_to_fov]
    )

    # define the mapping dict from manual to auto
    manual_to_auto_map = {}

    # compute the euclidean distances between the manual and the auto centroids
    manual_auto_dist = np.linalg.norm(
        manual_centroids[:, np.newaxis] - auto_centroids, axis=2
    )

    # for each manual point, get the index of the auto point closest to it
    closest_auto_point_ind = np.argmin(
        np.linalg.norm(manual_centroids[:, np.newaxis] - auto_centroids, axis=2),
        axis=1
    )

    # assign the mapping in manual_to_auto_map
    for manual_index, auto_index in enumerate(closest_auto_point_ind):
        # get the coordinates of the manual point and its corresponding closest auto point
        manual_coords = tuple(manual_centroids[manual_index])
        auto_coords = tuple(auto_centroids[auto_index])

        # use the coordinates as keys to get the fov names
        man_name = manual_centroid_to_fov[manual_coords]
        auto_name = auto_centroid_to_fov[auto_coords]

        # map the manual fov name to its closest auto fov name
        manual_to_auto_map[man_name] = auto_name

    return manual_to_auto_map, manual_fovs_info, auto_fovs_info


def generate_fov_circles(manual_to_auto_map, manual_fovs_info, auto_fovs_info,
                         manual_name, auto_name, slide_img, draw_radius=7):
    """Draw the circles defining each FOV (manually-specified and automatically-generated)

    Args:
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names
        manual_fovs_info (dict):
            maps each manual FOV to its centroid coordinates and size
        auto_fovs_info (dict):
            maps each automatically-generated FOV to its centroid coordinates and size
        manual_name (str):
            the name of the manual FOV to highlight
        auto_name (str):
            the name of the automatically-generated FOV to highlight
        slide_img (numpy.ndarray):
            the image to overlay
        draw_radius (int):
            the radius of the circle to overlay for each FOV, will be centered at the centroid

    Returns:
        numpy.ndarray:
            `slide_img` with defining each manually-defined and automatically-generated FOV
    """

    # define dicts to hold the coordinates
    manual_coords = {}
    auto_coords = {}

    # define the fov size boundaries, needed to prevent drawing a circle out of bounds
    fov_size = slide_img.shape[:2]

    # generate the regions for each manual and mapped auto fov
    for mfi in manual_fovs_info:
        # get the x- and y-coordinate of the centroid
        manual_x = int(manual_fovs_info[mfi][0])
        manual_y = int(manual_fovs_info[mfi][1])

        # define the circle coordinates for the region
        mr_x, mr_y = ellipse(
            manual_x, manual_y, draw_radius, draw_radius, shape=fov_size
        )

        # color the selected manual fov dark red, else bright red
        if mfi == manual_name:
            slide_img[mr_x, mr_y, 0] = 210
            slide_img[mr_x, mr_y, 1] = 37
            slide_img[mr_x, mr_y, 2] = 37
        else:
            slide_img[mr_x, mr_y, 0] = 255
            slide_img[mr_x, mr_y, 1] = 133
            slide_img[mr_x, mr_y, 2] = 133

    # repeat but for the automatically generated points
    for afi in auto_fovs_info:
        # repeat the above for auto points
        auto_x = int(auto_fovs_info[afi][0])
        auto_y = int(auto_fovs_info[afi][1])

        # define the circle coordinates for the region
        ar_x, ar_y = ellipse(
            auto_x, auto_y, draw_radius, draw_radius, shape=fov_size
        )

        # color the selected auto fov dark blue, else bright blue
        if afi == auto_name:
            slide_img[ar_x, ar_y, 0] = 50
            slide_img[ar_x, ar_y, 1] = 115
            slide_img[ar_x, ar_y, 2] = 229
        else:
            slide_img[ar_x, ar_y, 0] = 162
            slide_img[ar_x, ar_y, 1] = 197
            slide_img[ar_x, ar_y, 2] = 255

    return slide_img


def update_mapping_display(change, w_auto, manual_to_auto_map, manual_coords, auto_coords,
                           slide_img, draw_radius=7):
    """Changes the selected pairs of circles on the image based on new selected manual FOV

    Helper to `update_mapping` nested callback function in `interactive_remap`

    Args:
        change (dict):
            defines the properties of the changed value of the manual FOV menu
        w_auto (ipywidgets.widgets.widget_selection.Dropdown):
            the dropdown menu handler for the automatically-generated FOVs
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names
        manual_coords (dict):
            a `dict` defining each FOV in `manual_fov_regions` mapped to its centroid
            coordinates
        auto_coords (dict):
            a `dict` defining each FOV in `auto_fov_regions` mapped to its centroid
            coordinates
        slide_img (numpy.ndarray):
            the image to overlay
        draw_radius (int):
            the radius to draw each circle on the slide

    Returns:
        numpy.ndarray:
            `slide_img` with the updated circles after manual fov changed
    """

    # define the fov size boundaries, needed to prevent drawing a circle out of bounds
    fov_size = slide_img.shape[:2]

    # retrieve the old manual centroid
    old_manual_x, old_manual_y = manual_coords[change['old']]

    # redraw the old manual centroid on the slide_img
    old_mr_x, old_mr_y = ellipse(
        old_manual_x, old_manual_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[old_mr_x, old_mr_y, 0] = 255
    slide_img[old_mr_x, old_mr_y, 1] = 133
    slide_img[old_mr_x, old_mr_y, 2] = 133

    # retrieve the old auto centroid
    old_auto_x, old_auto_y = auto_coords[w_auto.value]

    # redraw the old auto centroid on the slide_img
    old_ar_x, old_ar_y = ellipse(
        old_auto_x, old_auto_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[old_ar_x, old_ar_y, 0] = 162
    slide_img[old_ar_x, old_ar_y, 1] = 197
    slide_img[old_ar_x, old_ar_y, 2] = 255

    # retrieve the new manual centroid
    new_manual_x, new_manual_y = manual_coords[change['new']]

    # redraw the new manual centroid on the slide_img
    new_mr_x, new_mr_y = ellipse(
        new_manual_x, new_manual_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[new_mr_x, new_mr_y, 0] = 210
    slide_img[new_mr_x, new_mr_y, 1] = 37
    slide_img[new_mr_x, new_mr_y, 2] = 37

    # retrieve the new auto centroid
    new_auto_x, new_auto_y = auto_coords[manual_to_auto_map[change['new']]]

    # redraw the new auto centroid on the slide_img
    new_ar_x, new_ar_y = ellipse(
        new_auto_x, new_auto_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[new_ar_x, new_ar_y, 0] = 50
    slide_img[new_ar_x, new_ar_y, 1] = 115
    slide_img[new_ar_x, new_ar_y, 2] = 229

    # set the mapped auto value according to the new manual value
    w_auto.value = manual_to_auto_map[change['new']]

    return slide_img


def remap_manual_to_auto_display(change, w_man, manual_to_auto_map, auto_coords,
                                 slide_img, draw_radius=7):
    """Changes the bolded automatically-generated FOV to new value selected for manual FOV
    and updates the mapping in `manual_to_auto_map`

    Helper to `remap_values` nested callback function in `interactive_remap`

    Args:
        change (dict):
            defines the properties of the changed value of the automatically-generated FOV menu
        w_man (ipywidgets.widgets.widget_selection.Dropdown):
            the dropdown menu handler for the manual FOVs
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names
        auto_coords (dict):
            maps each automatically-generated FOV to its annotation coordinate
        slide_img (numpy.ndarray):
            the image to overlay
        draw_radius (int):
            the radius to draw each circle on the slide

    Returns:
        numpy.ndarray:
            `slide_img` with the updated circles after auto fov changed remapping the fovs
    """

    # define the fov size boundaries, needed to prevent drawing a circle out of bounds
    fov_size = slide_img.shape[:2]

    # retrieve the coordinates for the old auto centroid w_prop mapped to
    old_auto_x, old_auto_y = auto_coords[change['old']]

    # redraw the old auto centroid on the slide_img
    old_ar_x, old_ar_y = ellipse(
        old_auto_x, old_auto_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[old_ar_x, old_ar_y, 0] = 162
    slide_img[old_ar_x, old_ar_y, 1] = 197
    slide_img[old_ar_x, old_ar_y, 2] = 255

    # retrieve the coordinates for the new auto centroid w_prop maps to
    new_auto_x, new_auto_y = auto_coords[change['new']]

    # redraw the new auto centroid on the slide_img
    new_ar_x, new_ar_y = ellipse(
        new_auto_x, new_auto_y, draw_radius, draw_radius, shape=fov_size
    )

    slide_img[new_ar_x, new_ar_y, 0] = 50
    slide_img[new_ar_x, new_ar_y, 1] = 115
    slide_img[new_ar_x, new_ar_y, 2] = 229

    # remap the manual fov to the changed value
    manual_to_auto_map[w_man.value] = change['new']

    return slide_img


def write_manual_to_auto_map(manual_to_auto_map, save_ann, mapping_path):
    """Saves the manually-defined to automatically-generated FOV map and notifies the user

    Helper to `save_mapping` nested callback function in `interactive_remap`

    Args:
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names
        save_ann (dict):
            contains the annotation object defining the save notification
        mapping_path (str):
            the path to the file to save the mapping to
    """

    # save the mapping
    with open(mapping_path, 'w') as mp:
        json.dump(manual_to_auto_map, mp)

    # remove the save annotation if it already exists
    # clears up some space if the user decides to save several times
    if save_ann['annotation']:
        save_ann['annotation'].remove()

    # display save annotation above the plot
    save_msg = plt.annotate(
        'Mapping saved!',
        (0, 20),
        color='white',
        fontweight='bold',
        annotation_clip=False
    )

    # assign annotation to save_ann
    save_ann['annotation'] = save_msg


def interactive_remap(manual_to_auto_map, manual_fovs_info,
                      auto_fovs_info, slide_img, mapping_path,
                      draw_radius=7, figsize=(7, 7)):
    """Creates the remapping interactive interface for manually-defined
    to automatically-generated FOVs

    Args:
        manual_to_auto_map (dict):
            defines the mapping of manual to auto FOV names
        manual_fovs_info (dict):
            maps each manual FOV to its centroid coordinates
        auto_fovs_info (dict):
            maps each automatically-generated FOV to its centroid coordinates
        slide_img (numpy.ndarray):
            the image to overlay
        mapping_path (str):
            the path to the file to save the mapping to
        draw_radius (int):
            the radius to draw each circle on the slide
        figsize (tuple):
            the size of the interactive figure to display
    """

    # error check: ensure mapping path exists
    if not os.path.exists(os.path.split(mapping_path)[0]):
        raise FileNotFoundError(
            "Path %s to mapping_path does not exist, "
            "please rename to a valid location" % os.path.split(mapping_path)[0]
        )

    # get the first manual fov, this will define the initial default value to display
    first_manual = list(manual_fovs_info.keys())[0]

    # define the two drop down menus, the first will define the manual fovs
    w_man = widgets.Dropdown(
        options=[mfi for mfi in list(manual_fovs_info.keys())],
        value=first_manual,
        description='Manually-defined FOV',
        layout=widgets.Layout(width='auto'),
        style={'description_width': 'initial'}
    )

    # the second will define the automatically-generated fovs
    # the default value should be set to the auto fov the initial manual fov maps to
    w_auto = widgets.Dropdown(
        options=[afi for afi in list(auto_fovs_info.keys())],
        value=manual_to_auto_map[first_manual],
        description='Automatically-generated FOV',
        layout=widgets.Layout(width='auto'),
        style={'description_width': 'initial'}
    )

    w_save = widgets.Button(
        description='Save mapping',
        layout=widgets.Layout(width='auto'),
        style={'description_width': 'initial'}
    )

    # define a box to hold w_man and w_auto
    w_box = widgets.HBox(
        [w_man, w_auto, w_save],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            align_items='stretch',
            width='75%'
        )
    )

    # display the box with w_man and w_auto dropdown menus
    display(w_box)

    # define an output context to display
    out = widgets.Output()

    # display the figure to plot on
    fig, ax = plt.subplots(figsize=figsize)

    # generate the circles and annotations for each circle to display on the image
    slide_img = generate_fov_circles(
        manual_to_auto_map, manual_fovs_info, auto_fovs_info, w_man.value, w_auto.value,
        slide_img, draw_radius
    )

    # make sure the output gets displayed to the output widget so it displays properly
    with out:
        # draw the image
        img_plot = ax.imshow(slide_img)

        # overwrite the default title
        _ = plt.title('Manually-defined to automatically-generated FOV map')

        # remove massive padding
        _ = plt.tight_layout()

        # define status of the save annotation, initially None, updates when user clicks w_save
        # NOTE: ipywidget callback functions can only access dicts defined in scope
        save_ann = {'annotation': None}

    # a callback function for changing w_auto to the value w_man maps to
    # NOTE: needs to be here so it can access w_man and w_auto in scope
    def update_mapping(change):
        """Updates `w_auto` and bolds a different manual-auto pair when `w_prop` changes

        Args:
            change (dict):
                defines the properties of the changed value in `w_prop`
        """

        # only operate if w_prop actually changed
        # prevents update if the user drops down w_prop but leaves it as the same manual fov
        if change['name'] == 'value' and change['new'] != change['old']:
            # need to be in the output widget context to update
            with out:
                new_slide_img = update_mapping_display(
                    change, w_auto, manual_to_auto_map, manual_fovs_info, auto_fovs_info,
                    slide_img, draw_radius
                )

                # set the new slide img in the plot
                img_plot.set_data(new_slide_img)
                fig.canvas.draw_idle()

    # a callback function for remapping when w_auto changes
    # NOTE: needs to be here so it can access w_man and w_auto in scope
    def remap_values(change):
        """Bolds the new `w_auto` and maps the selected FOV in `w_man`
        to the new `w_auto` in `manual_to_auto_amp`

        Args:
            change (dict):
                defines the properties of the changed value in `w_auto`
        """

        # only remap if the auto change as been updated
        # prevents update if the user drops down w_auto but leaves it as the same auto fov
        if change['name'] == 'value' and change['new'] != change['old']:
            # need to be in the output widget context to update
            with out:
                new_slide_img = remap_manual_to_auto_display(
                    change, w_man, manual_to_auto_map, auto_fovs_info, slide_img, draw_radius
                )

                # set the new slide img in the plot
                img_plot.set_data(new_slide_img)
                fig.canvas.draw_idle()

    # a callback function for saving manual_to_auto_map to mapping_path if w_save clicked
    def save_mapping(b):
        """Saves the mapping defined in `manual_to_auto_map`

        Args:
            b (ipywidgets.widgets.widget_button.Button):
                the button handler for `w_save`, only passed as a standard for `on_click` callback
        """

        # need to be in the output widget context to display status
        with out:
            # call the helper function to save manual_to_auto_map and notify user
            write_manual_to_auto_map(
                manual_to_auto_map, save_ann, mapping_path
            )

    # ensure a change to w_man redraws the image due to a new manual fov selected
    w_man.observe(update_mapping)

    # ensure a change to w_auto redraws the image due to a new automatic fov
    # mapped to the manual fov
    w_auto.observe(remap_values)

    # if w_save clicked, save the new mapping to the path defined in mapping_path
    w_save.on_click(save_mapping)

    # display the output
    display(out)


def remap_and_reorder_fovs(manual_fov_regions, manual_to_auto_map,
                           moly_path, randomize=False,
                           moly_insert=False, moly_interval=5):
    """Runs 3 separate tasks on `manual_fov_regions`:

    - Uses `manual_to_auto_map` to rename the FOVs
    - Randomizes the order of the FOVs if specified
    - Inserts Moly points at the specified interval if specified

    Args:
        manual_fov_regions (dict):
            The list of FOVs proposed by the user
        manual_to_auto_map (dict):
            Defines the mapping of manual to auto FOV names
        moly_path (str):
            The path to the Moly point to insert
        randomize (bool):
            Whether to randomize the FOVs
        moly_insert (bool):
            Whether to insert Moly points between FOVs at the specified `moly_interval`
        moly_interval (int):
            The interval in which to insert Moly points.
            Ignored if `moly_insert` is `False`.

    Returns:
        dict:
            `manual_fov_regions` with new FOV names, randomized, and with Moly points
    """

    # file path validation
    if not os.path.exists(moly_path):
        raise FileNotFoundError("Moly point %s does not exist" % moly_path)

    # load the Moly point in
    with open(moly_path, 'r') as mp:
        moly_point = json.load(mp)

    # error check: moly_interval cannot be less than or equal to 0
    if moly_interval <= 0:
        raise ValueError("moly_interval must be at least 1")

    # define a new fov regions dict for remapped names
    remapped_fov_regions = {}

    # copy over the metadata values from manual_fov_regions to remapped_fov_regions
    remapped_fov_regions = assign_metadata_vals(manual_fov_regions, remapped_fov_regions, ['fovs'])

    # define a new FOVs list for fov_regions_remapped
    remapped_fov_regions['fovs'] = []

    # rename the FOVs based on the mapping and append to fov_regions_remapped
    for fov in manual_fov_regions['fovs']:
        # needed to prevent early saving since interactive visualization cannot stop this
        # from running if a mapping_path provided already exists
        fov_data = copy.deepcopy(fov)

        # remap the name and append to fov_regions_remapped
        fov_data['name'] = manual_to_auto_map[fov['name']]
        remapped_fov_regions['fovs'].append(fov_data)

    # randomize the order of the FOVs if specified
    if randomize:
        remapped_fov_regions['fovs'] = shuffle(remapped_fov_regions['fovs'])

    # insert Moly points at the specified interval if specified
    if moly_insert:
        mi = moly_interval

        while mi < len(remapped_fov_regions['fovs']):
            remapped_fov_regions['fovs'].insert(mi, moly_point)
            mi += moly_interval + 1

    return remapped_fov_regions
