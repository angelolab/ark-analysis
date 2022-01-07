import copy
import datetime
from IPython.display import display, clear_output
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
        # read in the variable with correct dtype, print error message if cannot be coerced
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
    """Generate the region_params list in the tiling parameter dict

    Args:
        region_params (dict):
            A dictionary mapping each region-specific parameter to a list of values per fov

    Returns:
        list:
            The complete set of region_params sorted by run
    """

    # define the region params list
    region_params_list = []

    # iterate over all the region parameters, all parameter lists are the same length
    for i in range(len(region_params['region_start_x'])):
        # define a dict containing all the region info for the specific fov
        region_info = {
            rp: region_params[rp][i] for rp in region_params
        }

        # append info to region_params
        region_params_list.append(region_info)

    return region_params_list


def _read_tma_region_input(fov_tile_info, region_params):
    """Reads input for TMAs from user and fov_tile_info.

    Updates all the tiling params inplace. Units used are pixels.

    Args:
        fov_tile_info (dict):
            The data containing the fovs used to define each tiled region
        region_params (dict):
            A dictionary mapping each region-specific parameter to a list of values per fov
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
        if start_fov_x > end_fov_x:
            err_msg = ("Coordinate error for region %s: start x coordinates cannot be"
                       " greater than end coordinates")
            raise ValueError(err_msg % start_fov['name'])

        if start_fov_y < end_fov_y:
            err_msg = ("Coordinate error for region %s: start y coordinates cannot be"
                       " less than end coordinates")
            raise ValueError(err_msg % start_fov['name'])

        region_params['region_start_x'].append(start_fov_x)
        region_params['region_start_y'].append(start_fov_y)

        # the num_x, num_y, size_x, and size_y need additional validation
        # since they may not be compatible with the starting and ending coordinates
        while True:
            # allow the user to specify the number of fovs along each dimension
            num_x = read_tiling_param(
                "Enter number of x fovs for region %s (at least 3 required): " % start_fov['name'],
                "Error: number of x fovs must be a positive integer >=3",
                lambda nx: nx >= 3,
                dtype=int
            )

            num_y = read_tiling_param(
                "Enter number of y fovs for region %s (at least 3 required): " % start_fov['name'],
                "Error: number of y fovs must be a positive integer >=3",
                lambda ny: ny >= 3,
                dtype=int
            )

            # allow the user to specify the image size along each dimension
            size_x = read_tiling_param(
                "Enter the x image size for region %s (in microns): " % start_fov['name'],
                "Error: x step size must be a positive integer",
                lambda sx: sx >= 1,
                dtype=int
            )

            size_y = read_tiling_param(
                "Enter the y image size for region %s (in microns): " % start_fov['name'],
                "Error: y step size must be a positive integer",
                lambda sy: sy >= 1,
                dtype=int
            )

            # find num_x/num_y even intervals between start and end fov_x/fov_y
            # casted because indices cannot be floats
            # need .item() cast to prevent int64 is not JSON serializable error
            x_interval = [
                x.item() for x in np.linspace(start_fov_x, end_fov_x, num_x).astype(int)
            ]
            y_interval = list(reversed([
                y.item() for y in np.linspace(end_fov_y, start_fov_y, num_y).astype(int)
            ]))

            # get difference between x and y
            x_spacing = x_interval[1] - x_interval[0]
            y_spacing = y_interval[0] - y_interval[1]

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

        region_params['fov_num_x'].append(num_x)
        region_params['fov_num_y'].append(num_y)

        region_params['x_fov_size'].append(size_x)
        region_params['y_fov_size'].append(size_y)

        region_params['x_intervals'].append(list(x_interval))
        region_params['y_intervals'].append(list(y_interval))

        # allow the user to specify if the FOVs should be randomized
        randomize = read_tiling_param(
            "Randomize fovs for region %s? Y/N: " % start_fov['name'],
            "Error: randomize parameter must Y or N",
            lambda r: r in ['Y', 'N', 'y', 'n'],
            dtype=str
        )

        # make sure randomize is uppercase
        randomize = randomize.upper()

        region_params['region_rand'].append(randomize)


def _read_non_tma_region_input(fov_tile_info, region_params):
    """Reads input for non-TMAs from user and fov_tile_info

    Updates all the tiling params inplace. Units used are pixels.

    Args:
        fov_tile_info (dict):
            The data containing the fovs used to define each tiled region
        region_params (dict):
            A dictionary mapping each region-specific parameter to a list of values per fov
    """

    # read in the data for each fov (region_start from fov_list_path, fov_num from user)
    for fov in fov_tile_info['fovs']:
        region_params['region_start_x'].append(fov['centerPointMicrons']['x'])
        region_params['region_start_y'].append(fov['centerPointMicrons']['y'])

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

        region_params['fov_num_x'].append(num_x)
        region_params['fov_num_y'].append(num_y)

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

        region_params['x_fov_size'].append(size_x)
        region_params['y_fov_size'].append(size_y)

        # allow the user to specify if the FOVs should be randomized
        randomize = read_tiling_param(
            "Randomize fovs for region %s? Y/N: " % fov['name'],
            "Error: randomize parameter must Y or N",
            lambda r: r in ['Y', 'N', 'y', 'n'],
            dtype=str
        )

        randomize = randomize.upper()

        region_params['region_rand'].append(randomize)


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

    # define the region_params dict
    region_params = {rpf: [] for rpf in settings.REGION_PARAM_FIELDS}

    # remove x and y interval keys if non-TMA is used
    if not tma:
        region_params.pop('x_intervals')
        region_params.pop('y_intervals')

    # read in the tma inputs
    if tma:
        _read_tma_region_input(fov_tile_info, region_params)
    else:
        _read_non_tma_region_input(fov_tile_info, region_params)

    # need to copy fov metadata over, needed for create_tiled_regions
    tiling_params['fovs'] = copy.deepcopy(fov_tile_info['fovs'])

    # store the read in parameters in the region_params key
    tiling_params['region_params'] = generate_region_info(region_params)

    # whether to insert moly points between runs
    moly_run_insert = read_tiling_param(
        "Insert moly points between runs? Y/N: ",
        "Error: moly point run parameter must be either Y or N",
        lambda mri: mri in ['Y', 'N', 'y', 'n'],
        dtype=str
    )

    moly_run_insert = moly_run_insert.upper()

    tiling_params['moly_run'] = moly_run_insert

    # whether to insert moly points between tiles
    moly_interval_insert = read_tiling_param(
        "Specify moly point tile interval? Y/N: ",
        "Error: moly interval insertion parameter must either Y or N",
        lambda mii: mii in ['Y', 'N', 'y', 'n'],
        dtype=str
    )

    moly_interval_insert = moly_interval_insert.upper()

    # if moly insert is set, we need to specify an additional moly_interval param
    # NOTE: the interval applies regardless of if the tiles overlap runs or not
    if moly_interval_insert == 'Y':
        moly_interval = read_tiling_param(
            "Enter the fov interval size to insert moly points: ",
            "Error: moly interval must be a positive integer",
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


def generate_fov_list(tiling_params, moly_point, tma=False):
    """Generate the list of fovs on the image from the tiling_params set

    Args:
        tiling_params (dict):
            The tiling parameters created by set_tiling_params
        moly_point (dict):
            The moly point to insert between fovs (and intervals if specified in tiling_params)
        tma (bool):
            Whether the data in tiling_params is in TMA format or not

    Returns:
        dict:
            Data containing information about each fov, will be saved to JSON
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

    # iterate through each region and append created fovs to tiled_regions['fovs']
    for region_index, region_info in enumerate(tiling_params['region_params']):
        # extract start coordinates
        start_x = region_info['region_start_x']
        start_y = region_info['region_start_y']

        # generate range of x and y coordinates
        if tma:
            x_range = region_info['x_intervals']
            y_range = region_info['y_intervals']
        else:
            x_range = range(region_info['fov_num_x'])
            y_range = list(reversed(range(region_info['fov_num_y'])))

        # create all pairs between two lists
        x_y_pairs = generate_x_y_fov_pairs(x_range, y_range)

        # name the FOVs according to MIBI conventions
        fov_names = ['R%dC%d' % (y, x) for x in range(region_info['fov_num_x'])
                     for y in range(region_info['fov_num_y'])]

        # randomize pairs list if specified
        if region_info['region_rand'] == 'Y':
            # make sure the fov_names are set in the same shuffled indices for renaming
            x_y_pairs, fov_names = shuffle(x_y_pairs, fov_names)

        for index, (xi, yi) in enumerate(x_y_pairs):
            # set the current x and y coordinate
            if tma:
                cur_x = xi
                cur_y = yi
            else:
                cur_x = start_x + xi * region_info['x_fov_size']
                cur_y = start_y + yi * region_info['y_fov_size']

            # copy the fov metadata over and add cur_x, cur_y, and identifier
            fov = copy.deepcopy(tiling_params['fovs'][region_index])
            fov['centerPointMicrons']['x'] = cur_x
            fov['centerPointMicrons']['y'] = cur_y
            fov['name'] = fov_names[index]

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


def convert_microns_to_pixels(coord):
    """Convert the coordinate in stage microns to optical pixels

    In other words, co-register using the centroid of a FOV.

    The values are coerced to ints to allow indexing into the slide.
    Coordinates are also returned in (y, x) form to account for a different coordinate axis.

    Args:
        coord (tuple):
            The coordinate in microns to convert

    Returns:
        tuple:
            The converted coordinate from microns to pixels
    """

    # NOTE: all conversions are done using the fiducials
    # convert from microns to stage coordinates
    stage_coord_x = (coord[0] * 0.001001 - 0.3116)
    stage_coord_y = (coord[1] * 0.001018 - 0.6294)

    # convert from stage coordinates to pixels
    pixel_coord_x = (stage_coord_x + 27.79) / 0.06887
    pixel_coord_y = (stage_coord_y - 77.40) / -0.06926

    return (int(pixel_coord_y), int(pixel_coord_x))


def assign_closest_fovs(manual_fovs, auto_fovs, moly_point_name):
    """For each fov in tiled_regions_proposed, map it to its closest fov in tiled_regions_auto

    Args:
        manual_fovs (dict):
            The list of fovs proposed by the user.
            Assumed to have the same region dimension and the same Moly point
            as tiled_regions_auto.
        auto_fovs (dict):
            The list of fovs generated by `set_tiling_params` run in
            `example_fov_grid_generate.ipynb`
        moly_point_name (str):
            The name of the Moly point used in `tiled_regions_auto` and `tiled_regions_proposed`

    Returns:
        tuple:

        - A dict defining the mapping of fov names between `manual_fovs` and
          `auto_fovs`
        - A dict defining each fov in `manual_fovs` mapped to its centroid
          coordinates and size
        - A dict defining each fov in `auto_fovs` mapped to its centroid
          coordinates and size
    """

    # define the centroid and size info for manual_fovs and auto_fovs
    manual_fovs_info = {
        fov['name']: {
            'centroid': convert_microns_to_pixels(
                (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            ),
            'size': (fov['frameSizePixels']['width'], fov['frameSizePixels']['height'])
        }

        for fov in manual_fovs['fovs'] if fov['name'] != moly_point_name
    }

    auto_fovs_info = {
        fov['name']: {
            'centroid': convert_microns_to_pixels(
                (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            ),
            'size': (fov['frameSizePixels']['width'], fov['frameSizePixels']['height'])
        }

        for fov in auto_fovs['fovs'] if fov['name'] != moly_point_name
    }

    # we define these "reverse" dicts to map from centroid back to fov name
    # this makes it easier to use numpy broadcasting to help find the closest fov pairs
    manual_centroid_to_fov = {
        (manual_fovs_info[fov]['centroid'][0], manual_fovs_info[fov]['centroid'][1]): fov
        for fov in manual_fovs_info
    }

    auto_centroid_to_fov = {
        (auto_fovs_info[fov]['centroid'][0], auto_fovs_info[fov]['centroid'][1]): fov
        for fov in auto_fovs_info
    }

    # define numpy arrays of the proposed and auto centroids
    manual_centroids = np.array(
        [list(centroid) for centroid in manual_centroid_to_fov]
    )

    # define numpy arrays of the proposed and auto centroids
    auto_centroids = np.array(
        [list(centroid) for centroid in auto_centroid_to_fov]
    )

    # define the mapping dict from proposed to auto
    manual_to_auto_map = {}

    # compute the euclidean distances between the proposed and the auto centroids
    manual_auto_dist = np.linalg.norm(
        manual_centroids[:, np.newaxis] - auto_centroids, axis=2
    )

    # for each proposed point, get the index of the auto point closest to it
    closest_auto_point_ind = np.argmin(
        np.linalg.norm(manual_centroids[:, np.newaxis] - auto_centroids, axis=2),
        axis=1
    )

    # assign the mapping in proposed_to_auto_map
    for manual_index, auto_index in enumerate(closest_auto_point_ind):
        # get the coordinates of the proposed point and its corresponding closest auto point
        manual_coords = tuple(manual_centroids[manual_index])
        auto_coords = tuple(auto_centroids[auto_index])

        # use the coordinates as keys to get the fov names
        man_name = manual_centroid_to_fov[manual_coords]
        auto_name = auto_centroid_to_fov[auto_coords]

        # map the proposed fov name to its closest auto fov name
        manual_to_auto_map[man_name] = auto_name

    return manual_to_auto_map, manual_fovs_info, auto_fovs_info


def generate_fov_circles(manual_to_auto_map, manual_fovs_info, auto_fovs_info,
                         manual_name, auto_name, slide_img, draw_radius=5):
    """Draw the circles defining each fov (proposed and automatically-generated)

    Args:
        manual_to_auto_map (dict):
            defines the mapping of manual to auto fov names
        manual_fovs_info (dict):
            maps each manual fov to its centroid coordinates and size
        auto_fovs_info (dict):
            maps each automatically-generated fov to its centroid coordinates and size
        proposed_name (str):
            the name of the manual fov to highlight
        auto_name (str):
            the name of the automatically-generated fov to highlight
        slide_img (numpy.ndarray):
            the image to overlay
        draw_radius (int):
            the radius of the circle to overlay for each fov, will be centered at the centroid

    Returns:
        tuple:

        - A `numpy.ndarray` containing the slide_img with circles defining each fov
        - A dict mapping each manual fov to its annotation coordinate
        - A dict mapping each automatically-generated fov to its annotation coordinate
    """

    # define dictionaries to hold the coordinates
    manual_coords = {}
    auto_coords = {}

    # define the fov size boundaries, needed to prevent drawing a circle out of bounds
    fov_size = slide_img.shape[:2]

    # generate the regions for each proposed and mapped auto fov
    for mfi in manual_fovs_info:
        # get the x- and y-coordinate of the centroid
        manual_x = int(manual_fovs_info[mfi]['centroid'][0])
        manual_y = int(manual_fovs_info[mfi]['centroid'][1])

        # define the circle coordinates for the region
        mr_x, mr_y = ellipse(
            manual_x, manual_y, draw_radius, draw_radius, shape=fov_size
        )

        # color the selected proposed fov dark red, else bright red
        if mfi == manual_name:
            slide_img[mr_x, mr_y, 0] = 210
            slide_img[mr_x, mr_y, 1] = 37
            slide_img[mr_x, mr_y, 2] = 37
        else:
            slide_img[mr_x, mr_y, 0] = 255
            slide_img[mr_x, mr_y, 1] = 133
            slide_img[mr_x, mr_y, 2] = 133

        # define the annotations to place at each coordinate
        manual_coords[mfi] = (manual_x, manual_y)

    # repeat but for the automatically generated points
    for afi in auto_fovs_info:
        # repeat the above for auto points
        auto_x = int(auto_fovs_info[afi]['centroid'][0])
        auto_y = int(auto_fovs_info[afi]['centroid'][1])

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

        auto_coords[afi] = (auto_x, auto_y)

    return slide_img, manual_coords, auto_coords


def update_mapping_display(change, w_auto, manual_to_auto_map, manual_coords, auto_coords,
                           slide_img, draw_radius=7):
    """Changes the selected pairs of circles on the image based on new selected proposed fov

    Helper to `update_mapping` nested callback function in `interactive_remap`

    Args:
        change (dict):
            defines the properties of the changed value of the manual fov menu
        w_auto (ipywidgets.widgets.widget_selection.Dropdown):
            the dropdown menu handler for the automatically-generated fovs
        manual_to_auto_map (dict):
            defines the mapping of manual to auto fov names
        manual_coords (dict):
            a dict defining each fov in `fov_regions_proposed` mapped to its centroid
            coordinates
        auto_coords (dict):
            a dict defining each fov in `fov_regions_auto` mapped to its centroid
            coordinates
        slide_img (numpy.ndarray):
            the image to overlay
        draw_radius (int):
            the radius to draw each circle on the slide

    Returns:
        numpy.ndarray:
            the slide_img with the updated circles after proposed fov changed
    """

    # define the fov size boundaries, needed to prevent drawing a circle out of bounds
    fov_size = slide_img.shape[:2]

    # retrieve the old proposed centroid
    old_manual_x, old_manual_y = manual_coords[change['old']]

    # redraw the old proposed centroid on the slide_img
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

    # retrieve the new proposed centroid
    new_manual_x, new_manual_y = manual_coords[change['new']]

    # redraw the new proposed centroid on the slide_img
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

    # set the mapped auto value according to the new proposed value
    w_auto.value = manual_to_auto_map[change['new']]

    return slide_img


def remap_proposed_to_auto_display(change, w_man, manual_to_auto_map, auto_coords,
                                   slide_img, draw_radius=7):
    """Changes the bolded automatically-generated fov to new value selected for proposed fov
    and updates the mapping in proposed_to_auto_map

    Helper to `remap_values` nested callback function in `interactive_remap`

    Args:
        change (dict):
            defines the properties of the changed value of the automatically-generated fov menu
        w_man (ipywidgets.widgets.widget_selection.Dropdown):
            the dropdown menu handler for the manual fovs
        proposed_to_auto_map (dict):
            defines the mapping of proposed to auto fov names
        auto_coords (dict):
            maps each automatically-generated fov to its annotation coordinate
        slide_img (numpy.ndarray):
            the image to overlay
        draw_radius (int):
            the radius to draw each circle on the slide

    Returns:
        numpy.ndarray:
            the slide_img with the updated circles after auto fov changed remapping the fovs
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

    # remap the proposed fov to the changed value
    manual_to_auto_map[w_man.value] = change['new']

    return slide_img


def write_manual_to_auto_map(manual_to_auto_map, save_ann, mapping_path):
    """Saves the manually-defined to automatically-generated fov map and notifies the user

    Helper to `save_mapping` nested callback function in `interactive_remap`

    Args:
        manual_to_auto_map (dict):
            defines the mapping of manual to auto fov names
        save_ann (dict):
            contains the annotation object defining the save notification
        mapping_path (str):
            the path to the file to save the mapping to

    Returns:
        ipywidgets:
            the updated annotation contained in `save_ann`
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
    """Creates the remapping interactive interface

    Args:
        manual_to_auto_map (dict):
            defines the mapping of manual to auto fov names
        manual_fovs_info (dict):
            maps each manual fov to its centroid coordinates and size
        auto_fovs_info (dict):
            maps each automatically-generated fov to its centroid coordinates and size
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

    # get the first manual fov
    # this will define the initial default value to display
    first_manual = list(manual_fovs_info.keys())[0]

    # define the two drop down menus
    # the first will define the manual fovs
    w_man = widgets.Dropdown(
        options=[mfi for mfi in list(manual_fovs_info.keys())],
        value=first_manual,
        description='Manually-defined FOV',
        layout=widgets.Layout(width='auto'),
        style={'description_width': 'initial'}
    )

    # the second will define the automatically-generated fovs
    # the default value should be set to the auto fov the initial proposed fov maps to
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
    slide_img, manual_coords, auto_coords = generate_fov_circles(
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
    # NOTE: needs to be here so it can easily access w_man and w_auto
    def update_mapping(change):
        """Updates w_auto and bolds a different proposed-auto pair when w_prop changes

        Args:
            change (dict):
                defines the properties of the changed value in w_prop
        """

        # only operate if w_prop actually changed
        # prevents update if the user drops down w_prop but leaves it as the same proposed fov
        if change['name'] == 'value' and change['new'] != change['old']:
            # need to be in the output widget context to update
            with out:
                # call the helper function to redraw circles on the slide_img
                new_slide_img = update_mapping_display(
                    change, w_auto, manual_to_auto_map, manual_coords, auto_coords,
                    slide_img, draw_radius
                )

                # set the new slide img in the plot
                img_plot.set_data(new_slide_img)
                fig.canvas.draw_idle()

    # a callback function for remapping when w_auto changes
    # NOTE: needs to be here so it can easily access w_man and w_auto
    def remap_values(change):
        """Bolds the new w_auto and maps the selected fov in w_man
        to the new w_auto in `proposed_to_auto_amp`

        Args:
            change (dict):
                defines the properties of the changed value in w_auto
        """

        # only remap if the auto change as been updated
        # prevents update if the user drops down w_auto but leaves it as the same auto fov
        if change['name'] == 'value' and change['new'] != change['old']:
            # need to be in the output widget context to update
            with out:
                # call the helper function to redraw the circles on the slide_img
                # and update proposed_to_auto_map with the new w_prop mapping
                new_slide_img = remap_proposed_to_auto_display(
                    change, w_man, manual_to_auto_map, auto_coords, slide_img, draw_radius
                )

                # set the new slide img in the plot
                img_plot.set_data(new_slide_img)
                fig.canvas.draw_idle()

    # a callback function for saving proposed_to_auto_map to mapping_path if w_save clicked
    def save_mapping(b):
        """Saves the mapping defined in `proposed_to_auto_map`

        Args:
            b (ipywidgets.widgets.widget_button.Button):
                the button handler for w_save, only passed as a standard for `on_click` callback
        """

        # need to be in the output widget context to display status
        with out:
            # call the helper function to save proposed_to_auto_map and notify user
            write_manual_to_auto_map(
                manual_to_auto_map, save_ann, mapping_path
            )

    # ensure a change to w_man redraws the image due to a new proposed fov selected
    w_man.observe(update_mapping)

    # ensure a change to w_auto redraws the image due to a new automatic fov
    # mapped to the proposed fov
    w_auto.observe(remap_values)

    # if w_save clicked, save the new mapping to the path defined in mapping_path
    w_save.on_click(save_mapping)

    # display the output
    display(out)


def remap_and_reorder_fovs(manual_fov_regions, manual_to_auto_map,
                           moly_point, randomize=False,
                           moly_insert=False, moly_interval=5):
    """Runs 3 separate tasks on `fov_regions_proposed`:

    - Uses manual_to_auto_map to rename the FOVs
    - Randomizes the order of the FOVs if specified
    - Inserts Moly points at the specified interval if specified

    Args:
        manual_fov_regions (dict):
            The list of fovs proposed by the user
        proposed_to_auto_map (dict):
            Defines the mapping of proposed to auto fov names
        moly_point (dict):
            The Moly point to insert
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

    # error check: moly_interval cannot be less than or equal to 0
    if moly_interval <= 0:
        raise ValueError("moly_interval must be at least 1")

    # define a new fov regions proposed dict
    remapped_fov_regions = {}

    # get the metadata values to copy over
    metadata_keys = list(manual_fov_regions.keys())
    metadata_keys.remove('fovs')

    # for simplicity's sake, copy over only the string, int, float, and bool values
    for mk in metadata_keys:
        if type(manual_fov_regions[mk]) in [str, int, float, bool]:
            remapped_fov_regions[mk] = manual_fov_regions[mk]

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
        for mi in range(moly_interval, len(remapped_fov_regions['fovs']), moly_interval + 1):
            remapped_fov_regions['fovs'].insert(mi, moly_point)

    return remapped_fov_regions
