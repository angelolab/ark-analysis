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
        if start_fov_x > end_fov_x or start_fov_y > end_fov_y:
            err_msg = ("Coordinate error for region %s: start coordinates cannot be"
                       " greater than end coordinates")
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
            x_interval = [x.item() for x in np.linspace(start_fov_x, end_fov_x, num_x).astype(int)]
            y_interval = [y.item() for y in np.linspace(start_fov_y, end_fov_y, num_y).astype(int)]

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


def create_tiled_regions(tiling_params, moly_point, tma=False):
    """Create the tiled regions for each fov

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
            x_range = region_info['x_intervals']
            y_range = region_info['y_intervals']
        else:
            x_range = list(range(region_info['fov_num_x']))
            y_range = list(range(region_info['fov_num_y']))

        # create all pairs between two lists
        x_y_pairs = generate_x_y_fov_pairs(x_range, y_range)

        # name the FOVs according to MIBI conventions
        fov_names = ['R%dC%d' % (x, y) for y in range(region_info['fov_num_y'])
                     for x in reversed(range(region_info['fov_num_x']))]

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


def assign_closest_tiled_regions(tiled_regions_proposed, tiled_regions_auto, moly_point_name):
    """For each tile in tiled_regions_proposed, map it to its closest tile in tiled_regions_auto

    Args:
        tiled_regions_proposed (dict):
            The list of tiles proposed by the user.
            Assumed to have the same tiled region dimension and the same Moly point
            as tiled_regions_auto.
        tiled_regions_auto (dict):
            The list of tiles generated by `set_tiling_params` run in
            `example_fov_grid_generate.ipynb`
        moly_point_name (str):
            The name of the Moly point used in `tiled_regions_auto` and `tiled_regions_proposed`

    Returns:
        tuple:

        - A dict defining the mapping of tile names between `tiled_regions_proposed` and
          `tiled_regions_auto`
        - A dict defining each tile in `tiled_regions_proposed` mapped to its centroid
          coordinates and size
        - A dict defining each tile in `tiled_regions_auto` mapped to its centroid
          coordinates and size
    """

    # define the centroid and size info for tiled_regions_proposed and tiled_regions_auto
    proposed_tiles_info = {
        fov['name']: {
            'centroid': convert_microns_to_pixels(
                (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            ),
            'size': (fov['frameSizePixels']['width'], fov['frameSizePixels']['height'])
        }

        for fov in tiled_regions_proposed['fovs'] if fov['name'] != moly_point_name
    }

    auto_tiles_info = {
        fov['name']: {
            'centroid': convert_microns_to_pixels(
                (fov['centerPointMicrons']['x'], fov['centerPointMicrons']['y'])
            ),
            'size': (fov['frameSizePixels']['width'], fov['frameSizePixels']['height'])
        }

        for fov in tiled_regions_auto['fovs'] if fov['name'] != moly_point_name
    }

    # we define these "reverse" dicts to map from centroid back to tile name
    # this makes it easier to use numpy broadcasting to help find the closest tile pairs
    proposed_centroid_to_tile = {
        (proposed_tiles_info[fov]['centroid'][0], proposed_tiles_info[fov]['centroid'][1]): fov
        for fov in proposed_tiles_info
    }

    auto_centroid_to_tile = {
        (auto_tiles_info[fov]['centroid'][0], auto_tiles_info[fov]['centroid'][1]): fov
        for fov in auto_tiles_info
    }

    # define numpy arrays of the proposed and auto centroids
    proposed_centroids = np.array(
        [list(centroid) for centroid in proposed_centroid_to_tile]
    )

    # define numpy arrays of the proposed and auto centroids
    auto_centroids = np.array(
        [list(centroid) for centroid in auto_centroid_to_tile]
    )

    # define the mapping dict from proposed to auto
    proposed_to_auto_map = {}

    # compute the euclidean distances between the proposed and the auto centroids
    proposed_auto_dist = np.linalg.norm(
        proposed_centroids[:, np.newaxis] - auto_centroids, axis=2
    )

    # for each proposed point, get the index of the auto point closest to it
    closest_auto_point_ind = np.argmin(
        np.linalg.norm(proposed_centroids[:, np.newaxis] - auto_centroids, axis=2),
        axis=1
    )

    # assign the mapping in proposed_to_auto_map
    for prop_index, auto_index in enumerate(closest_auto_point_ind):
        # get the coordinates of the proposed point and its corresponding closest auto point
        prop_coords = tuple(proposed_centroids[prop_index])
        auto_coords = tuple(auto_centroids[auto_index])

        # use the coordinates as keys to get the tile names
        prop_name = proposed_centroid_to_tile[prop_coords]
        auto_name = auto_centroid_to_tile[auto_coords]

        # map the proposed tile name to its closest auto tile name
        proposed_to_auto_map[prop_name] = auto_name

    return proposed_to_auto_map, proposed_tiles_info, auto_tiles_info


def generate_tile_circles(proposed_to_auto_map, proposed_tiles_info, auto_tiles_info,
                          slide_img, draw_radius=50):
    """Draw the circles defining each tile (proposed and automatically-generated)

    Args:
        proposed_to_auto_map (dict):
            defines the mapping of proposed to auto tile names
        proposed_tiles_info (dict):
            maps each proposed tile to its centroid coordinates and size
        auto_tiles_info (dict):
            maps each automatically-generated tile to its centroid coordinates and size
        slide_img (numpy.ndarray):
            the image to overlay
        draw_radius (int):
            the radius of the circle to overlay for each tile, will be centered at the centroid

    Returns:
        tuple:

        - A numpy.ndarray containing the slide_img with circles defining each tile
        - A dict mapping each proposed tile to its annotation coordinate
        - A dict mapping each automatically-generated tile to its annotation coordinate
    """

    # define dictionaries to hold the annotations
    proposed_annot = {}
    auto_annot = {}

    # define the tile size boundaries, needed to prevent drawing a circle out of bounds
    tile_size = slide_img.shape[:2]

    # generate the regions for each proposed and mapped auto tile
    for pti in proposed_tiles_info:
        # get the x- and y-coordinate of the centroid
        proposed_x = int(proposed_tiles_info[pti]['centroid'][0])
        proposed_y = int(proposed_tiles_info[pti]['centroid'][1])

        # define the circle coordinates for the region
        pr_x, pr_y = ellipse(
            proposed_x, proposed_y, draw_radius, draw_radius, shape=tile_size
        )

        # color each tile red for proposed
        slide_img[pr_x, pr_y, 0] = 238
        slide_img[pr_x, pr_y, 1] = 75
        slide_img[pr_x, pr_y, 2] = 43

        # define the annotations to place at each coordinate
        proposed_annot[pti] = (proposed_x, proposed_y)

    # repeat but for the automatically generated points
    for ati in auto_tiles_info:
        # repeat the above for auto points
        auto_x = int(auto_tiles_info[ati]['centroid'][0])
        auto_y = int(auto_tiles_info[ati]['centroid'][1])

        # define the circle coordinates for the region
        ar_x, ar_y = ellipse(
            auto_x, auto_y, draw_radius, draw_radius, shape=tile_size
        )

        # color each tile blue for auto
        slide_img[ar_x, ar_y, 0] = 135
        slide_img[ar_x, ar_y, 1] = 206
        slide_img[ar_x, ar_y, 2] = 250

        auto_annot[ati] = (auto_x, auto_y)

    return slide_img, proposed_annot, auto_annot


def generate_tile_annotations(proposed_annot, auto_annot, proposed_name, auto_name):
    """Generates the initial set of annotations to display over each tile

    Args:
        proposed_annot (dict):
            maps each proposed tile to its annotation
        auto_annot (dict):
            maps each automatically-generated tile to its annotation
        proposed_name (str):
            the name of the proposed tile to highlight
        auto_name (str):
            the name of the automatically-generated tile to highlight

    Returns:
        tuple:

        - A dict mapping each proposed tile to its matplotlib annotation object
        - A dict mapping each automatically-generated tile to its matplotlib annotation object
    """

    # define dicts to store the proposed and the auto annotations
    pa_anns = {}
    aa_anns = {}

    # generate the annotation text for proposed tiles
    for pa_text, pa_coord in proposed_annot.items():
        # increase size of and bold the proposed tile name if that's the one selected
        font_weight = 'bold' if pa_text == proposed_name else 'normal'
        font_color = 'green' if pa_text == proposed_name else 'white'

        # draw the proposed tile name
        pa_ann = plt.annotate(
            pa_text,
            (pa_coord[1], pa_coord[0]),
            color=font_color,
            ha='center',
            fontweight=font_weight,
            fontsize=5
        )

        # add annotation to pa_anns
        pa_anns[pa_text] = pa_ann

    # generate the annotation text for auto tiles
    for aa_text, aa_coord in auto_annot.items():
        # increase size of and bold the auto tile name if that's the one selected
        font_weight = 'bold' if aa_text == auto_name else 'normal'
        font_color = 'green' if aa_text == auto_name else 'black'

        # draw the auto tile name
        aa_ann = plt.annotate(
            aa_text,
            (aa_coord[1], aa_coord[0]),
            color=font_color,
            ha='center',
            fontweight=font_weight,
            fontsize=5
        )

        # add annotation to aa_anns
        aa_anns[aa_text] = aa_ann

    return pa_anns, aa_anns


def update_mapping_display(change, proposed_to_auto_map, proposed_annot, auto_annot,
                           pa_anns, aa_anns, w_auto):
    """Changes the bolded proposed to automatically-generated tile in the remapping visualization
    based on change in proposed tile menu

    Helper to `update_mapping` nested callback function in `interactive_remap`

    Args:
        change (dict):
            defines the properties of the changed value of the proposed tile menu
        proposed_to_auto_map (dict):
            defines the mapping of proposed to auto tile names
        proposed_annot (dict):
            maps each proposed tile to its annotation coordinate
        auto_annot (dict):
            maps each automatically-generated tile to its annotation coordinate
        pa_anns (dict):
            maps each proposed tile to its matplotlib annotation object
        aa_anns (dict):
            maps each automatically-generated tile to its matplotlib annotation object
        w_auto (ipywidgets.widgets.widget_selection.Dropdown):
            the dropdown menu handler for the automatically-generated tiles
    """

    # remove annotations for the proposed annotations and the new auto annotation
    pa_anns[change['old']].remove()
    pa_anns[change['new']].remove()
    aa_anns[proposed_to_auto_map[change['new']]].remove()

    # only remove the old auto annotation if it doesn't match the new one
    # otherwise we'll be removing the same text twice and it will fail
    if w_auto.value != proposed_to_auto_map[change['new']]:
        aa_anns[w_auto.value].remove()

    # create a new unbolded annotation for the proposed tile
    old_pa_ann = plt.annotate(
        change['old'],
        (proposed_annot[change['old']][1], proposed_annot[change['old']][0]),
        color='white',
        ha='center',
        fontweight='normal',
        fontsize=5
    )

    # update the annotation for the old proposed tile
    pa_anns[change['old']] = old_pa_ann

    # only create a new unbolded annotation for the old auto tile if it doesn't
    # match the new one, otherwise both a normal and bold name will be drawn
    if w_auto.value != proposed_to_auto_map[change['new']]:
        old_aa_ann = plt.annotate(
            w_auto.value,
            (auto_annot[w_auto.value][1], auto_annot[w_auto.value][0]),
            color='black',
            ha='center',
            fontweight='normal',
            fontsize=5
        )

        # update the annotation for the old auto tile
        aa_anns[w_auto.value] = old_aa_ann

    # create a new bolded annotation for the new pair
    new_pa_ann = plt.annotate(
        change['new'],
        (proposed_annot[change['new']][1], proposed_annot[change['new']][0]),
        color='green',
        ha='center',
        fontweight='bold',
        fontsize=5
    )

    # update the annotation for the new proposed tile
    pa_anns[change['new']] = new_pa_ann

    new_aa_ann = plt.annotate(
        proposed_to_auto_map[change['new']],
        (
            auto_annot[proposed_to_auto_map[change['new']]][1],
            auto_annot[proposed_to_auto_map[change['new']]][0]
        ),
        color='green',
        ha='center',
        fontweight='bold',
        fontsize=5
    )

    # update the annotation for the new auto tile
    aa_anns[proposed_to_auto_map[change['new']]] = new_aa_ann

    # set the mapped auto value according to the new proposed value
    w_auto.value = proposed_to_auto_map[change['new']]


def remap_proposed_to_auto_display(change, proposed_to_auto_map, auto_annot, aa_anns, w_prop):
    """Changes the bolded automatically-generated tile to new value selected for proposed tile
    and updates the mapping in proposed_to_auto_map

    Helper to `remap_values` nested callback function in `interactive_remap`

    Args:
        change (dict):
            defines the properties of the changed value of the automatically-generated tile menu
        proposed_to_auto_map (dict):
            defines the mapping of proposed to auto tile names
        auto_annot (dict):
            maps each automatically-generated tile to its annotation coordinate
        aa_anns (dict):
            maps each automatically-generated tile to its matplotlib annotation object
        w_prop (ipywidgets.widgets.widget_selection.Dropdown):
            the dropdown menu handler for the proposed tiles
    """

    # remove annotation for the old and new value w_prop maps to
    aa_anns[change['old']].remove()
    aa_anns[change['new']].remove()

    # generate a new un-bolded annotation for the old value w_prop mapped to
    old_aa_ann = plt.annotate(
        change['old'],
        (auto_annot[change['old']][1], auto_annot[change['old']][0]),
        color='black',
        ha='center',
        fontweight='normal',
        fontsize=5
    )

    # update the annotation of the old value w_prop mapped to
    aa_anns[change['old']] = old_aa_ann

    # generate a new bolded annotation for the new value w_prop mapped to
    new_aa_ann = plt.annotate(
        change['new'],
        (auto_annot[change['new']][1], auto_annot[change['new']][0]),
        color='green',
        ha='center',
        fontweight='bold',
        fontsize=5
    )

    # update the annotation of the new value w_prop maps to
    aa_anns[change['new']] = new_aa_ann

    # remap the proposed tile to the changed value
    proposed_to_auto_map[w_prop.value] = change['new']


def write_proposed_to_auto_map(proposed_to_auto_map, save_ann, mapping_path):
    """Saves the proposed to automatically-generated tile map and notifies the user

    Helper to `save_mapping` nested callback function in `interactive_remap`

    Args:
        proposed_to_auto_map (dict):
            defines the mapping of proposed to auto tile names
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
        json.dump(proposed_to_auto_map, mp)

    # remove the save annotation if it already exists
    # clears up some space if the user decides to save several times
    if save_ann['annotation']:
        save_ann['annotation'].remove()

    # display save annotation above the plot
    save_msg = plt.annotate(
        'Mapping saved!',
        (0, -5),
        color='black',
        fontweight='bold',
        annotation_clip=False
    )

    # assign annotation to save_ann
    save_ann['annotation'] = save_msg


def interactive_remap(proposed_to_auto_map, proposed_tiles_info,
                      auto_tiles_info, slide_img, mapping_path,
                      draw_radius=5, figsize=(15, 15)):
    """Creates the remapping interactive interface

    Args:
        proposed_to_auto_map (dict):
            defines the mapping of proposed to auto tile names
        proposed_tiles_info (dict):
            maps each proposed tile to its centroid coordinates and size
        auto_tiles_info (dict):
            maps each automatically-generated tile to its centroid coordinates and size
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

    # get the first proposed tile
    # this will define the initial default value to display
    first_proposed = list(proposed_tiles_info.keys())[0]

    # define the two drop down menus
    # the first will define the proposed tile mappings
    w_prop = widgets.Dropdown(
        options=[pti for pti in list(proposed_tiles_info.keys())],
        value=first_proposed,
        description='Proposed tile',
        layout=widgets.Layout(width='auto'),
        style={'description_width': 'initial'}
    )

    # the second will define the automatically-generated tile mappings
    # the default value should be set to the auto tile the initial proposed tile maps to
    w_auto = widgets.Dropdown(
        options=[ati for ati in list(auto_tiles_info.keys())],
        value=proposed_to_auto_map[first_proposed],
        description='Automatically-generated tile',
        layout=widgets.Layout(width='auto'),
        style={'description_width': 'initial'}
    )

    w_save = widgets.Button(
        description='Save mapping',
        layout=widgets.Layout(width='auto'),
        style={'description_width': 'initial'}
    )

    # define a box to hold w_prop and w_auto
    w_box = widgets.HBox(
        [w_prop, w_auto, w_save],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row',
            align_items='stretch',
            width='75%'
        )
    )

    # display the box with w_prop and w_auto dropdown menus
    display(w_box)

    # define an output context to display
    out = widgets.Output()

    # display the figure to plot on
    fig, ax = plt.subplots(figsize=figsize)

    # generate the circles and annotations for each circle to display on the image
    slide_img, proposed_annot, auto_annot = generate_tile_circles(
        proposed_to_auto_map, proposed_tiles_info, auto_tiles_info, slide_img, draw_radius
    )

    # make sure the output gets displayed to the output widget so it displays properly
    with out:
        # draw the image
        img_plot = ax.imshow(slide_img)

        # overwrite the default title
        _ = plt.title('Proposed to automatically-generated tile map')

        # remove massive padding
        _ = plt.tight_layout()

        # generate the tile annotations
        pa_anns, aa_anns = generate_tile_annotations(
            proposed_annot, auto_annot, w_prop.value, w_auto.value
        )

        # define status of the save annotation, nitially None, updates when user clicks w_save
        # NOTE: ipywidget callback functions can only access dicts defined in scope
        save_ann = {'annotation': None}

    # a callback function for changing w_auto to the value w_prop maps to
    # NOTE: needs to be here so it can easily access w_prop and w_auto
    def update_mapping(change):
        """Updates w_auto and bolds a different proposed-auto pair when w_prop changes

        Args:
            change (dict):
                defines the properties of the changed value in w_prop
        """

        # only operate if w_prop actually changed
        # prevents update if the user drops down w_prop but leaves it as the same proposed tile
        if change['name'] == 'value' and change['new'] != change['old']:
            # need to be in the output widget context to update
            with out:
                # call the helper function to update annotations
                update_mapping_display(
                    change, proposed_to_auto_map, proposed_annot, auto_annot,
                    pa_anns, aa_anns, w_auto
                )

    # a callback function for remapping when w_auto changes
    # NOTE: needs to be here so it can easily access w_prop and w_auto
    def remap_values(change):
        """Bolds the new w_auto and maps the selected tile in w_prop
        to the new w_auto in `proposed_to_auto_amp`

        Args:
            change (dict):
                defines the properties of the changed value in w_auto
        """

        # only remap if the auto change as been updated
        # prevents update if the user drops down w_auto but leaves it as the same auto tile
        if change['name'] == 'value' and change['new'] != change['old']:
            # need to be in the output widget context to update
            with out:
                # call the helper function to remap and update annotations
                remap_proposed_to_auto_display(
                    change, proposed_to_auto_map, auto_annot, aa_anns, w_prop
                )

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
            write_proposed_to_auto_map(
                proposed_to_auto_map, save_ann, mapping_path
            )

    # ensure a change to w_prop redraws the image due to a new proposed tile selected
    w_prop.observe(update_mapping)

    # ensure a change to w_auto redraws the image due to a new automatic tile
    # mapped to the proposed tile
    w_auto.observe(remap_values)

    # if w_save clicked, save the new mapping to the path defined in mapping_path
    w_save.on_click(save_mapping)

    # display the output
    display(out)
