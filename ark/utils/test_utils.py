import os
from copy import deepcopy
from random import choices
from string import ascii_lowercase

import numpy as np
import pandas as pd
import skimage.io as io
import xarray as xr

import ark.settings as settings
from ark.utils import synthetic_spatial_datagen
from ark.utils.tiff_utils import write_mibitiff


def _make_blank_file(folder, name):
    with open(os.path.join(folder, name), 'w'):
        pass


def gen_fov_chan_names(num_fovs, num_chans, return_imgs=False, use_delimiter=False):
    """Generate fov and channel names

    Names have the format 'fov0', 'fov1', ..., 'fovN' for fovs and 'chan0', 'chan1', ...,
    'chanM' for channels.

    Args:
        num_fovs (int):
            Number of fov names to create
        num_chans (int):
            Number of channel names to create
        return_imgs (bool):
            Return 'chanK.tiff' as well if True.  Default is False
        use_delimiter (bool):
            Appends '_otherinfo' to the first fov. Useful for testing fov id extraction from
            filenames.  Default is False

    Returns:
        tuple (list, list) or (list, list, list):
            If return_imgs is False, only fov and channel names are returned
            If return_imgs is True, image names will also be returned
    """
    fovs = [f'fov{i}' for i in range(num_fovs)]
    if use_delimiter:
        fovs[0] = f'{fovs[0]}_otherinfo'
    chans = [f'chan{i}' for i in range(num_chans)]

    if return_imgs:
        imgs = [f'{chan}.tiff' for chan in chans]
        return fovs, chans, imgs
    else:
        return fovs, chans


# required metadata for mibitiff writing
MIBITIFF_METADATA = {
    'run': '20180703_1234_test', 'date': '2017-09-16T15:26:00',
    'coordinates': (12345, -67890), 'size': 500., 'slide': '857',
    'fov_id': 'fov1', 'fov_name': 'R1C3_Tonsil',
    'folder': 'fov1/RowNumber0/Depth_Profile0',
    'dwell': 4, 'scans': '0,5', 'aperture': 'B',
    'instrument': 'MIBIscope1', 'tissue': 'Tonsil',
    'panel': '20170916_1x', 'mass_offset': 0.1, 'mass_gain': 0.2,
    'time_resolution': 0.5, 'miscalibrated': False, 'check_reg': False,
    'filename': '20180703_1234_test', 'description': 'test image',
    'version': 'alpha',
}


def _gen_tif_data(fov_number, chan_number, img_shape, fills, dtype):
    """Generates random or set-filled image data

    Args:
        fov_number (int):
            Number of fovs required
        chan_number (int):
            Number of channels required
        img_shape (tuple):
            Single image dimensions (x pixels, y pixels)
        fills (bool):
            If False, data is randomized.  If True, each single image will be filled with a value
            one less than that of the next channel.  If said image is the last channel, then the
            value is one less than that of the first channel in the next fov.
        dtype (type):
            Data type for generated data

    Returns:
        numpy.ndarray:
            Image data with shape (fov_number, img_shape[0], img_shape[1], chan_number)

    """
    if not fills:
        tif_data = np.random.randint(0, 100,
                                     size=(fov_number, *img_shape, chan_number)).astype(dtype)
    else:
        tif_data = np.full(
            (*img_shape, fov_number, chan_number),
            (np.arange(fov_number * chan_number) % 256).reshape(fov_number, chan_number),
            dtype=dtype
        )
        tif_data = np.moveaxis(tif_data, 2, 0)

    return tif_data


def _gen_label_data(fov_number, comp_number, img_shape, dtype):
    """Generates quadrant-based label data

    Args:
        fov_number (int):
            Number of fovs required
        comp_number (int):
            Number of components
        img_shape (tuple):
            Single image dimensions (x pixels, y pixesl)
        dtype (type):
            Data type for generated labels

    Returns:
        numpy.ndarray:
            Label data with shape (fov_number, img_shape[0], img_shape[1], comp_number)
    """
    label_data = np.zeros((fov_number, *img_shape, comp_number), dtype=dtype)

    right = (img_shape[1] - 1) // 2
    left = (img_shape[1] + 2) // 2
    up = (img_shape[0] - 1) // 2
    down = (img_shape[0] + 2) // 2

    counter = 1
    for fov in range(fov_number):
        for comp in range(comp_number):
            label_data[fov, :up, :right, comp] = counter
            counter = (counter % 255) + 1
            label_data[fov, :up, left:, comp] = counter
            counter = (counter % 255) + 1
            label_data[fov, down:, :right, comp] = counter
            counter = (counter % 255) + 1
            label_data[fov, down:, left:, comp] = counter
            counter = (counter % 255) + 1

    return label_data


def _write_tifs(base_dir, fov_names, img_names, shape, sub_dir, fills, dtype):
    """Generates and writes single tifs to into base_dir/fov_name/sub_dir

    Args:
        base_dir (str):
            Path to base directory
        fov_names (list):
            List of fov folders to create/fill
        img_names (list):
            Channel names
        shape (tuple):
            Single image shape (x pixels, y pixels)
        sub_dir (str):
            Subdirectory to write images into
        fills (bool):
            If False, data is randomized.  If True, each single image will be filled with a value
            one less than that of the next channel.  If said image is the last channel, then the
            value is one less than that of the first channel in the next fov.
        dtype (type):
            Data type for generated images

    Returns:
        tuple (dict, numpy.ndarray):

        - File locations, indexable by fov names
        - Image data as an array with shape (num_fovs, shape[0], shape[1], num_channels)
    """
    tif_data = _gen_tif_data(len(fov_names), len(img_names), shape, fills, dtype)

    if sub_dir is None:
        sub_dir = ""

    filelocs = {}

    for i, fov in enumerate(fov_names):
        filelocs[fov] = []
        fov_path = os.path.join(base_dir, fov, sub_dir)
        os.makedirs(fov_path)
        for j, name in enumerate(img_names):
            io.imsave(os.path.join(fov_path, f'{name}.tiff'), tif_data[i, :, :, j],
                      check_contrast=False)
            filelocs[fov].append(os.path.join(fov_path, name))

    return filelocs, tif_data


def _write_multitiff(base_dir, fov_names, channel_names, shape, sub_dir, fills,
                     dtype, channels_first=False):
    """Generates and writes multitifs to into base_dir

    Args:
        base_dir (str):
            Path to base directory
        fov_names (list):
            List of fov files to write
        channel_names (list):
            Channel names
        shape (tuple):
            Single image shape (x pixels, y pixels)
        sub_dir (str):
            Ignored.
        fills (bool):
            If False, data is randomized.  If True, each single image will be filled with a value
            one less than that of the next channel.  If said image is the last channel, then the
            value is one less than that of the first channel in the next fov.
        dtype (type):
            Data type for generated images
        channels_first(bool):
            Indicates whether the data should be saved in channels_first format. Default: False

    Returns:
        tuple (dict, numpy.ndarray):

        - File locations, indexable by fov names
        - Image data as an array with shape (num_fovs, shape[0], shape[1], num_channels)
    """
    tif_data = _gen_tif_data(len(fov_names), len(channel_names), shape, fills, dtype)

    filelocs = {}

    for i, fov in enumerate(fov_names):
        tiffpath = os.path.join(base_dir, f'{fov}.tiff')
        v = tif_data[i, :, :, :]
        if channels_first:
            v = np.moveaxis(v, -1, 0)
        io.imsave(tiffpath, v, plugin='tifffile',
                  check_contrast=False)
        filelocs[fov] = tiffpath

    return filelocs, tif_data


def _write_mibitiff(base_dir, fov_names, channel_names, shape, sub_dir, fills, dtype):
    """Generates and writes mibitiffs to into base_dir

    Args:
        base_dir (str):
            Path to base directory
        fov_names (list):
            List of fov files to write
        channel_names (list):
            Channel names
        shape (tuple):
            Single image shape (x pixels, y pixels)
        sub_dir (str):
            Ignored.
        fills (bool):
            If False, data is randomized.  If True, each single image will be filled with a value
            one less than that of the next channel.  If said image is the last channel, then the
            value is one less than that of the first channel in the next fov.
        dtype (type):
            Data type for generated images

    Returns:
        tuple (dict, numpy.ndarray):

        - File locations, indexable by fov names
        - Image data as an array with shape (num_fovs, shape[0], shape[1], num_channels)
    """
    tif_data = _gen_tif_data(len(fov_names), len(channel_names), shape, fills, dtype)

    filelocs = {}

    mass_map = tuple(enumerate(channel_names, 1))

    for i, fov in enumerate(fov_names):
        tiffpath = os.path.join(base_dir, f'{fov}.tiff')
        write_mibitiff(tiffpath, tif_data[i, :, :, :], mass_map, MIBITIFF_METADATA)
        filelocs[fov] = tiffpath

    return filelocs, tif_data


def _write_reverse_multitiff(base_dir, fov_names, channel_names, shape, sub_dir, fills, dtype):
    """Generates and writes 'reversed' multitifs to into base_dir

    Saved images have shape (num_channels, shape[0], shape[1]).  This is mostly useful for
    testing deepcell-input loading.

    Args:
        base_dir (str):
            Path to base directory
        fov_names (list):
            List of fov files to write
        channel_names (list):
            Channel names
        shape (tuple):
            Single image shape (x pixels, y pixels)
        sub_dir (str):
            Ignored.
        fills (bool):
            If False, data is randomized.  If True, each single image will be filled with a value
            one less than that of the next channel.  If said image is the last channel, then the
            value is one less than that of the first channel in the next fov.
        dtype (type):
            Data type for generated images

    Returns:
        tuple (dict, numpy.ndarray):

        - File locations, indexable by fov names
        - Image data as an array with shape (num_fovs, shape[0], shape[1], num_channels)
    """
    tif_data = _gen_tif_data(len(channel_names), len(fov_names), shape, fills, dtype)

    filelocs = {}

    for i, fov in enumerate(fov_names):
        tiffpath = os.path.join(base_dir, f'{fov}.tiff')
        io.imsave(tiffpath, tif_data[:, :, :, i], plugin='tifffile',
                  check_contrast=False)
        filelocs[fov] = tiffpath

    tif_data = np.swapaxes(tif_data, 0, -1)

    return filelocs, tif_data


def _write_labels(base_dir, fov_names, comp_names, shape, sub_dir, fills, dtype):
    """Generates and writes label maps to into base_dir

    Args:
        base_dir (str):
            Path to base directory
        fov_names (list):
            List of fov files to write
        comp_names (list):
            Component names
        shape (tuple):
            Single image shape (x pixels, y pixels)
        sub_dir (str):
            Ignored.
        fills (bool):
            Ignored.
        dtype (type):
            Data type for generated labels

    Returns:
        tuple (dict, numpy.ndarray):

        - File locations, indexable by fov names
        - Label data as an array with shape (num_fovs, shape[0], shape[1], num_components)
    """
    label_data = _gen_label_data(len(fov_names), len(comp_names), shape, dtype)

    filelocs = {}

    for i, fov in enumerate(fov_names):
        tiffpath = os.path.join(base_dir, f'{fov}.tiff')
        io.imsave(tiffpath, label_data[i, :, :, 0], plugin='tifffile',
                  check_contrast=False)
        filelocs[fov] = tiffpath

    return filelocs, label_data


TIFFMAKERS = {
    'tiff': _write_tifs,
    'multitiff': _write_multitiff,
    'reverse_multitiff': _write_reverse_multitiff,
    'mibitiff': _write_mibitiff,
    'labels': _write_labels,
}


def create_paired_xarray_fovs(base_dir, fov_names, channel_names, img_shape=(10, 10),
                              mode='tiff', delimiter=None, sub_dir=None, fills=False,
                              dtype="int8", channels_first=False):
    """Writes data to file system (images or labels) and creates expected xarray for reloading
    data from said file system.

    Args:
        base_dir (str):
            Path to base directory.  All data will be written into this folder.
        fov_names (list):
            List of fovs
        channel_names (list):
            List of channels/components
        img_shape (tuple):
            Single image shape (x pixels, y pixels)
        mode (str):
            The type of data to generate. Current options are:

            - 'tiff'
            - 'multitiff'
            - 'reverse_multitiff'
            - 'mibitiff'
            - 'labels'
        delimiter (str or None):
            Delimiting character or string separating fov_id from rest of file/folder name.
            Default is None.
        sub_dir (str):
            Only active for 'tiff' mode.  Creates another sub directory in which tiffs are stored
            within the parent fov folder.  Default is None.
        fills (bool):
            Only active for image data (not 'labels'). If False, data is randomized.  If True,
            each single image will be filled with a value one less than that of the next channel.
            If said image is the last channel, then the value is one less than that of the first
            channel in the next fov.
        dtype (type):
            Data type for generated images/labels.  Default is int16
        channels_first (bool):
            Indicates whether the data should be saved in channels_first format when
            mode is 'multitiff'. Default: False

    Returns:
        tuple (dict, xarray.DataArray):

        - File locations, indexable by fov names
        - Image/label data as an xarray with shape
          (num_fovs, im_shape[0], shape[1], num_channels)
    """

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f'{base_dir} is not a directory')

    if fov_names is None or fov_names is []:
        raise ValueError('No fov names were given...')

    if channel_names is None or channel_names is []:
        raise ValueError('No image names were given...')

    if not isinstance(fov_names, list):
        fov_names = [fov_names]

    if not isinstance(channel_names, list):
        channel_names = [channel_names]

    if mode == 'multitiff':
        filelocs, tif_data = TIFFMAKERS[mode](base_dir, fov_names, channel_names,
                                              img_shape, sub_dir, fills, dtype,
                                              channels_first=channels_first)
    else:
        filelocs, tif_data = TIFFMAKERS[mode](base_dir, fov_names, channel_names,
                                              img_shape, sub_dir, fills, dtype)

    if delimiter is not None:
        fov_ids = [fov.split(delimiter)[0] for fov in fov_names]
    else:
        fov_ids = fov_names

    if 'multitiff' in mode:
        channel_names = range(len(channel_names))

    if mode == 'labels':
        data_xr = make_labels_xarray(tif_data, fov_ids, channel_names, *img_shape)
    else:
        data_xr = make_images_xarray(tif_data, fov_ids, channel_names, *img_shape)

    return filelocs, data_xr


def make_images_xarray(tif_data, fov_ids=None, channel_names=None, row_size=10, col_size=10,
                       dtype='int16'):
    """Generate a correctly formatted image data xarray

    Args:
        tif_data (numpy.ndarray or None):
            Image data to embed within the xarray.  If None, randomly generated image data is used,
            but fov_ids and channel_names must not be None.
        fov_ids (list or None):
            List of fov names.  If None, fov id's will be generated based on the shape of tif_data
            following the scheme 'fov0', 'fov1', ... , 'fovN'. Default is None.
        channel_names (list or None):
            List of channel names.  If None, channel names will be generated based on the shape of
            tif_data following the scheme 'chan0', 'chan1', ... , 'chanM'.  Default is None.
        row_size (int):
            Horizontal size of individual image.  Default is 10.
        col_size (int):
            Vertical size of individual image. Default is 10.
        dtype (type):
            Data type for generated images.  Default is int16.

    Returns:
        xarray.DataArray:
            Image data with standard formatting
    """
    if tif_data is None:
        tif_data = _gen_tif_data(len(fov_ids), len(channel_names), (row_size, col_size), False,
                                 dtype=dtype)
    else:
        row_size, col_size = tif_data.shape[1:3]

        buf_fov_ids, buf_chan_names = gen_fov_chan_names(tif_data.shape[0], tif_data.shape[-1])
        if fov_ids is None:
            fov_ids = buf_fov_ids
        if channel_names is None:
            channel_names = buf_chan_names

    coords = [fov_ids, range(row_size), range(col_size), channel_names]
    dims = ["fovs", "rows", "cols", "channels"]
    return xr.DataArray(tif_data, coords=coords, dims=dims)


def make_labels_xarray(label_data, fov_ids=None, compartment_names=None, row_size=10, col_size=10,
                       dtype='int16'):
    """Generate a correctly formatted label data xarray

    Args:
        label_data (numpy.ndarray or None):
            Label data to embed within the xarray.  If None, automatically generated label data is
            used, but fov_ids and compartment_names must not be None.
        fov_ids (list or None):
            List of fov names.  If None, fov id's will be generated based on the shape of tif_data
            following the scheme 'fov0', 'fov1', ... , 'fovN'. Default is None.
        compartment_names (list or None):
            List of compartment names.  If None, compartment names will be ['whole_cell'] or
            ['whole_cell', 'nuclear'] if label_data.shape[-1] is 1 or 2 respecticely. Default is
            None.
        row_size (int):
            Horizontal size of individual image.  Default is 10.
        col_size (int):
            Vertical size of individual image. Default is 10.
        dtype (type):
            Data type for generated labels.  Default is int16.

    Returns:
        xarray.DataArray:
            Label data with standard formatting
    """
    if label_data is None:
        label_data = _gen_label_data(len(fov_ids), len(compartment_names), (row_size, col_size),
                                     dtype=dtype)
    else:
        row_size, col_size = label_data.shape[1:3]

        buf_fov_ids, _ = gen_fov_chan_names(label_data.shape[0], 0)
        if fov_ids is None:
            fov_ids = buf_fov_ids
        if compartment_names is None:
            comp_dict = {1: ['whole_cell'], 2: ['whole_cell', 'nuclear']}
            compartment_names = comp_dict[label_data.shape[-1]]

    coords = [fov_ids, range(row_size), range(col_size), compartment_names]
    dims = ['fovs', 'rows', 'cols', 'compartments']
    return xr.DataArray(label_data, coords=coords, dims=dims)


TEST_MARKERS = list('ABCDEFG')


def make_cell_table(num_cells, extra_cols=None):
    """ Generate a cell table with default column names for testing purposes.

    Args:
        num_cells (int):
            Number of rows (cells) in the cell table
        extra_cols (dict):
            Extra columns to add in the format ``{'Column_Name' : data_1D, ...}``

    Returns:
        pandas.DataFrame:
            A structural example of a cell table containing simulated marker expressions,
            cluster labels, centroid coordinates, and more.

    """
    # columns from regionprops extraction
    region_cols = [x for x in settings.REGIONPROPS_BASE if
                   x not in ['label', 'area', 'centroid']] + settings.REGIONPROPS_SINGLE_COMP
    region_cols += settings.REGIONPROPS_MULTI_COMP
    # consistent ordering of column names
    column_names = [settings.FOV_ID,
                    settings.PATIENT_ID,
                    settings.CLUSTER_ID,
                    settings.KMEANS_CLUSTER,
                    settings.CELL_LABEL,
                    settings.CELL_TYPE,
                    settings.CELL_SIZE] + TEST_MARKERS + region_cols + ['centroid-0', 'centroid-1']

    if extra_cols is not None:
        column_names += list(extra_cols.values())

    # random filler data
    cell_data = pd.DataFrame(np.random.random(size=(num_cells, len(column_names))),
                             columns=column_names)
    # not-so-random filler data
    cluster_id = choices(range(1, 21), k=num_cells)
    centroids = pd.DataFrame(np.array([(x, y) for x in range(1024) for y in range(1024)]))
    centroid_loc = np.random.choice(range(1024 ** 2), size=num_cells, replace=False)
    fields = [(settings.FOV_ID, choices(range(1, 5), k=num_cells)),
              (settings.PATIENT_ID, choices(range(1, 10), k=num_cells)),
              (settings.CLUSTER_ID, cluster_id),
              (settings.KMEANS_CLUSTER, [ascii_lowercase[i] for i in cluster_id]),
              (settings.CELL_LABEL, list(range(num_cells))),
              (settings.CELL_TYPE, choices(ascii_lowercase, k=num_cells)),
              (settings.CELL_SIZE, np.random.uniform(100, 300, size=num_cells)),
              (settings.CENTROID_0, np.array(centroids.iloc[centroid_loc, 0])),
              (settings.CENTROID_1, np.array(centroids.iloc[centroid_loc, 1]))
              ]

    for name, col in fields:
        cell_data[name] = col

    return cell_data


# TODO: Use these below


EXCLUDE_CHANNELS = [
    "Background",
    "HH3",
    "summed_channel",
]

DEFAULT_COLUMNS_LIST = \
    [settings.CELL_SIZE] \
    + list(range(1, 24)) \
    + [
        settings.CELL_LABEL,
        'area',
        'eccentricity',
        'maj_axis_length',
        'min_axis_length',
        'perimiter',
        settings.FOV_ID,
        settings.CLUSTER_ID,
        settings.CELL_TYPE,
    ]
list(map(
    DEFAULT_COLUMNS_LIST.__setitem__, [1, 14, 23], EXCLUDE_CHANNELS
))

DEFAULT_COLUMNS = dict(zip(range(33), DEFAULT_COLUMNS_LIST))


def create_test_extraction_data():
    """Generate hardcoded extraction test data

    Returns:
        tuple (numpy.ndarray, numpy.ndarray):

        - a sample segmentation mask
        - sample corresponding channel data
    """
    # first create segmentation masks
    cell_mask = np.zeros((1, 40, 40, 1), dtype='int16')
    cell_mask[:, 4:10, 4:10:, :] = 1
    cell_mask[:, 15:25, 20:30, :] = 2
    cell_mask[:, 27:32, 3:28, :] = 3
    cell_mask[:, 35:40, 15:22, :] = 5

    # then create channels data
    channel_data = np.zeros((1, 40, 40, 5), dtype="int16")
    channel_data[:, :, :, 0] = 1
    channel_data[:, :, :, 1] = 5
    channel_data[:, :, :, 2] = 5
    channel_data[:, :, :, 3] = 10
    channel_data[:, :, :, 4] = 0

    # cell1 is the only cell negative for channel 3
    channel_data[:, 4:10, 4:10, 3] = 0

    # cell2 is the only cell positive for channel 4
    channel_data[:, 15:25, 20:30, 4] = 10

    return cell_mask, channel_data


def _make_neighborhood_matrix():
    """Generate a sample neighborhood matrix

    Returns:
        pandas.DataFrame:
            a sample neighborhood matrix with three different populations,
            intended to test clustering
    """
    col_names = {0: settings.FOV_ID, 1: settings.CELL_LABEL, 2: 'feature1', 3: 'feature2'}
    neighbor_counts = pd.DataFrame(np.zeros((200, 5)))
    neighbor_counts = neighbor_counts.rename(col_names, axis=1)

    neighbor_counts.iloc[0:100, 0] = "fov1"
    neighbor_counts.iloc[0:100, 1] = np.arange(100) + 1
    neighbor_counts.iloc[0:50, 2:4] = np.random.randint(low=0, high=10, size=(50, 2))
    neighbor_counts.iloc[50:100, 2:4] = np.random.randint(low=990, high=1000, size=(50, 2))

    neighbor_counts.iloc[100:200, 0] = "fov2"
    neighbor_counts.iloc[100:200, 1] = np.arange(100) + 1
    neighbor_counts.iloc[100:150, 2:4] = np.random.randint(low=990, high=1000, size=(50, 2))
    neighbor_counts.iloc[150:200, 2] = np.random.randint(low=0, high=10, size=50)
    neighbor_counts.iloc[150:200, 3] = np.random.randint(low=990, high=1000, size=50)

    return neighbor_counts


# TODO: it's very clunky and confusing to have to separate spatial analysis
# from spatial analysis utils synthetic data generation, here's an example
# of a function that I'd like to see be shared across both testing modules
# in the future
def _make_threshold_mat(in_utils):
    """Generate sample marker thresholds for testing channel enrichment

    Args:
        in_utils (bool):
            whether to generate for spatial_analysis or spatial_analysis_utils testing

    Returns:
        pandas.DataFrame:
            a sample marker threshold matrix for thresholding specifically for channel enrichment
    """

    thresh = pd.DataFrame(np.zeros((20, 2)))
    thresh.iloc[:, 1] = .5

    if not in_utils:
        thresh.iloc[:, 0] = np.concatenate([np.arange(2, 14), np.arange(15, 23)])

        # spatial analysis should still be correct regardless of the marker threshold ordering
        thresh = thresh.sample(frac=1).reset_index(drop=True)

    return thresh


def _make_dist_mat_sa(enrichment_type, dist_lim):
    """Generate a sample distance matrix to test spatial_analysis

    Args:
        enrichment_type (str):
            whether to generate for positive, negative, or no enrichment
        dist_lim (int):
            the threshold to use for selecting entries in the distance matrix for enrichment

    Returns:
        xarray.DataArray:
            a sample distance matrix to use for testing spatial_analysis
    """

    if enrichment_type not in ["none", "positive", "negative"]:
        raise ValueError("enrichment_type must be none, positive, or negative")

    if enrichment_type == "none":
        # Create a 60 x 60 euclidian distance matrix of random values for no enrichment
        np.random.seed(0)
        rand_mat = np.random.randint(0, 200, size=(60, 60))
        np.fill_diagonal(rand_mat[:, :], 0)

        rand_mat = xr.DataArray(rand_mat,
                                coords=[np.arange(rand_mat.shape[0]) + 1,
                                        np.arange(rand_mat.shape[1]) + 1])

        fovs = ["fov8", "fov9"]
        mats = [rand_mat, rand_mat]
        rand_matrix = dict(zip(fovs, mats))

        return rand_matrix
    elif enrichment_type == "positive":
        # Create positive enrichment distance matrix where 10 cells mostly positive for marker 1
        # are located close in proximity to 10 cells mostly positive for marker 2.
        # Other included cells are not significantly positive for either marker and are located
        # far from the two positive populations.

        dist_mat_pos = synthetic_spatial_datagen.generate_test_dist_matrix(
            num_A=10, num_B=10, num_C=60, distr_AB=(int(dist_lim / 5), 1),
            distr_random=(int(dist_lim * 5), 1)
        )

        fovs = ["fov8", "fov9"]
        mats = [dist_mat_pos, dist_mat_pos]
        dist_mat_pos = dict(zip(fovs, mats))

        return dist_mat_pos
    elif enrichment_type == "negative":
        # This creates a distance matrix where there are two groups of cells significant for 2
        # different markers that are not located near each other (not within the dist_lim).

        dist_mat_neg = synthetic_spatial_datagen.generate_test_dist_matrix(
            num_A=20, num_B=20, num_C=20, distr_AB=(int(dist_lim * 5), 1),
            distr_random=(int(dist_lim / 5), 1)
        )

        fovs = ["fov8", "fov9"]
        mats = [dist_mat_neg, dist_mat_neg]
        dist_mat_neg = dict(zip(fovs, mats))

        return dist_mat_neg


def _make_expression_mat_sa(enrichment_type):
    """Generate a sample expression matrix to test spatial_analysis

    Args:
        enrichment_type (str):
            whether to generate for positive, negative, or no enrichment

    Returns:
        pandas.DataFrame:
            an expression matrix with cell labels and patient labels
    """

    if enrichment_type not in ["none", "positive", "negative"]:
        raise ValueError("enrichment_type must be none, positive, or negative")

    if enrichment_type == "none":
        all_data = pd.DataFrame(np.zeros((120, 33)))
        # Assigning values to the patient label and cell label columns
        # We create data for two fovs, with the second fov being the same as the first but the
        # cell expression data for marker 1 and marker 2 are inverted. cells 0-59 are fov8 and
        # cells 60-119 are fov9
        all_data.loc[0:59, 30] = "fov8"
        all_data.loc[60:, 30] = "fov9"
        all_data.loc[0:59, 24] = np.arange(60) + 1
        all_data.loc[60:, 24] = np.arange(60) + 1
        # We create two populations of 20 cells, each positive for different marker (index 2 and 3)
        all_data.iloc[0:20, 2] = 1
        all_data.iloc[20:40, 3] = 1

        all_data.iloc[60:80, 3] = 1
        all_data.iloc[80:100, 2] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data.iloc[0:20, 31] = 1
        all_data.iloc[0:20, 32] = "Pheno1"
        all_data.iloc[60:80, 31] = 2
        all_data.iloc[60:80, 32] = "Pheno2"

        all_data.iloc[20:40, 31] = 2
        all_data.iloc[20:40, 32] = "Pheno2"
        all_data.iloc[80:100, 31] = 1
        all_data.iloc[80:100, 32] = "Pheno1"

        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data = all_data.rename(DEFAULT_COLUMNS, axis=1)

        all_patient_data.loc[all_patient_data.iloc[:, 31] == 0, settings.CELL_TYPE] = "Pheno3"
        return all_patient_data
    elif enrichment_type == "positive":
        all_data_pos = pd.DataFrame(np.zeros((160, 33)))
        # Assigning values to the patient label and cell label columns
        all_data_pos.loc[0:79, 30] = "fov8"
        all_data_pos.loc[80:, 30] = "fov9"
        all_data_pos.loc[0:79, 24] = np.arange(80) + 1
        all_data_pos.loc[80:, 24] = np.arange(80) + 1
        # We create 8 cells positive for column index 2, and 8 cells positive for column index 3.
        # These are within the dist_lim in dist_mat_pos (positive enrichment distance matrix).
        all_data_pos.iloc[0:8, 2] = 1
        all_data_pos.iloc[10:18, 3] = 1

        all_data_pos.iloc[80:88, 3] = 1
        all_data_pos.iloc[90:98, 2] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data_pos.iloc[0:8, 31] = 1
        all_data_pos.iloc[0:8, 32] = "Pheno1"
        all_data_pos.iloc[80:88, 31] = 2
        all_data_pos.iloc[80:88, 32] = "Pheno2"

        all_data_pos.iloc[10:18, 31] = 2
        all_data_pos.iloc[10:18, 32] = "Pheno2"
        all_data_pos.iloc[90:98, 31] = 1
        all_data_pos.iloc[90:98, 32] = "Pheno1"
        # We create 4 cells in column index 2 and column index 3 that are also positive
        # for their respective markers.
        all_data_pos.iloc[28:32, 2] = 1
        all_data_pos.iloc[32:36, 3] = 1
        all_data_pos.iloc[108:112, 3] = 1
        all_data_pos.iloc[112:116, 2] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data_pos.iloc[28:32, 31] = 1
        all_data_pos.iloc[28:32, 32] = "Pheno1"
        all_data_pos.iloc[108:112, 31] = 2
        all_data_pos.iloc[108:112, 32] = "Pheno2"

        all_data_pos.iloc[32:36, 31] = 2
        all_data_pos.iloc[32:36, 32] = "Pheno2"
        all_data_pos.iloc[112:116, 31] = 1
        all_data_pos.iloc[112:116, 32] = "Pheno1"

        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_pos = all_data_pos.rename(DEFAULT_COLUMNS, axis=1)

        all_patient_data_pos.loc[all_patient_data_pos.iloc[:, 31] == 0,
                                 settings.CELL_TYPE] = "Pheno3"
        return all_patient_data_pos
    elif enrichment_type == "negative":
        all_data_neg = pd.DataFrame(np.zeros((120, 33)))
        # Assigning values to the patient label and cell label columns
        all_data_neg.loc[0:59, 30] = "fov8"
        all_data_neg.loc[60:, 30] = "fov9"
        all_data_neg.loc[0:59, 24] = np.arange(60) + 1
        all_data_neg.loc[60:, 24] = np.arange(60) + 1
        # We create two groups of 20 cells positive for marker 1 (in column index 2)
        # and marker 2 (in column index 3) respectively.
        # The two populations are not within the dist_lim in dist_mat_neg
        all_data_neg.iloc[0:20, 2] = 1
        all_data_neg.iloc[20:40, 3] = 1

        all_data_neg.iloc[60:80, 3] = 1
        all_data_neg.iloc[80:100, 2] = 1
        # We assign the two populations of cells different cell phenotypes
        all_data_neg.iloc[0:20, 31] = 1
        all_data_neg.iloc[0:20, 32] = "Pheno1"
        all_data_neg.iloc[60:80, 31] = 2
        all_data_neg.iloc[60:80, 32] = "Pheno2"

        all_data_neg.iloc[20:40, 31] = 2
        all_data_neg.iloc[20:40, 32] = "Pheno2"
        all_data_neg.iloc[80:100, 31] = 1
        all_data_neg.iloc[80:100, 32] = "Pheno1"

        # Assign column names to columns not for markers (columns to be excluded)
        all_patient_data_neg = all_data_neg.rename(DEFAULT_COLUMNS, axis=1)

        all_patient_data_neg.loc[all_patient_data_neg.iloc[:, 31] == 0,
                                 settings.CELL_TYPE] = "Pheno3"
        return all_patient_data_neg


def _make_dist_exp_mats_spatial_test(enrichment_type, dist_lim):
    """Generate example expression and distance matrices for testing spatial_analysis

    Args:
        enrichment_type (str):
            whether to generate for positive, negative, or no enrichment
        dist_lim (int):
            the threshold to use for selecting entries in the distance matrix for enrichment

    Returns:
        tuple (pandas.DataFrame, xarray.DataArray):

        - a sample expression matrix
        - a sample distance matrix
    """

    all_data = _make_expression_mat_sa(enrichment_type=enrichment_type)
    dist_mat = _make_dist_mat_sa(enrichment_type=enrichment_type, dist_lim=dist_lim)

    return all_data, dist_mat


def _make_dist_mat_sa_utils():
    """Generate a sample distance matrix to test spatial_analysis_utils

    Returns:
        xarray.DataArray:
            a sample distance matrix to use for testing spatial_analysis_utils
    """

    dist_mat = np.zeros((10, 10))
    np.fill_diagonal(dist_mat, 0)

    # Create distance matrix where cells positive for marker 1 and 2 are within the dist_lim of
    # each other, but not the other groups. This is repeated for cells positive for marker 3 and 4,
    # and for cells positive for marker 5.
    dist_mat[1:4, 0] = 50
    dist_mat[0, 1:4] = 50
    dist_mat[4:9, 0] = 200
    dist_mat[0, 4:9] = 200
    dist_mat[9, 0] = 500
    dist_mat[0, 9] = 500
    dist_mat[2:4, 1] = 50
    dist_mat[1, 2:4] = 50
    dist_mat[4:9, 1] = 150
    dist_mat[1, 4:9] = 150
    dist_mat[9, 1:9] = 200
    dist_mat[1:9, 9] = 200
    dist_mat[3, 2] = 50
    dist_mat[2, 3] = 50
    dist_mat[4:9, 2] = 150
    dist_mat[2, 4:9] = 150
    dist_mat[4:9, 3] = 150
    dist_mat[3, 4:9] = 150
    dist_mat[5:9, 4] = 50
    dist_mat[4, 5:9] = 50
    dist_mat[6:9, 5] = 50
    dist_mat[5, 6:9] = 50
    dist_mat[7:9, 6] = 50
    dist_mat[6, 7:9] = 50
    dist_mat[8, 7] = 50
    dist_mat[7, 8] = 50

    # add some randomization to the ordering
    coords_in_order = np.arange(dist_mat.shape[0])
    coords_permuted = deepcopy(coords_in_order)
    np.random.shuffle(coords_permuted)
    dist_mat = dist_mat[np.ix_(coords_permuted, coords_permuted)]

    # we have to 1-index coords because people will be labeling their cells 1-indexed
    coords_dist_mat = [coords_permuted + 1, coords_permuted + 1]
    dist_mat = xr.DataArray(dist_mat, coords=coords_dist_mat)

    return dist_mat


def _make_expression_mat_sa_utils():
    """Generate a sample expression matrix to test spatial_analysis_utils

    Returns:
        pandas.DataFrame:
            an expression matrix with cell labels and patient labels
    """

    # Create example all_patient_data cell expression matrix
    all_data = pd.DataFrame(np.zeros((10, 33)))

    # Assigning values to the patient label and cell label columns
    all_data[30] = "fov8"
    all_data[24] = np.arange(len(all_data[1])) + 1

    colnames = {
        0: settings.CELL_SIZE,
        24: settings.CELL_LABEL,
        30: settings.FOV_ID,
        31: settings.CLUSTER_ID,
        32: settings.CELL_TYPE
    }
    all_data = all_data.rename(colnames, axis=1)

    # Create 4 cells positive for marker 1 and 2, 5 cells positive for markers 3 and 4,
    # and 1 cell positive for marker 5
    all_data.iloc[0:4, 2] = 1
    all_data.iloc[0:4, 3] = 1
    all_data.iloc[4:9, 5] = 1
    all_data.iloc[4:9, 6] = 1
    all_data.iloc[9, 7] = 1
    all_data.iloc[9, 8] = 1

    # 4 cells assigned one phenotype, 5 cells assigned another phenotype,
    # and the last cell assigned a different phenotype
    all_data.iloc[0:4, 31] = 1
    all_data.iloc[0:4, 32] = "Pheno1"
    all_data.iloc[4:9, 31] = 2
    all_data.iloc[4:9, 32] = "Pheno2"
    all_data.iloc[9, 31] = 3
    all_data.iloc[9, 32] = "Pheno3"

    return all_data


def _make_dist_exp_mats_spatial_utils_test():
    """Generate example expression and distance matrices for testing spatial_analysis_utils

    Returns:
        tuple (pandas.DataFrame, xarray.DataArray):

        - a sample expression matrix
        - a sample distance matrix
    """

    all_data = _make_expression_mat_sa_utils()
    dist_mat = _make_dist_mat_sa_utils()

    return all_data, dist_mat


def generate_sample_fov_tiling_entry(coord, name):
    """Generates a sample fov entry to put in a sample fovs list for tiling

    Args:
        coord (tuple):
            Defines the starting x and y point for the fov
        name (str):
            Defines the name of the fov

    Returns:
        dict:
            An entry to be placed in the fovs list with provided coordinate and name
    """

    sample_fov_tiling_entry = {
        "scanCount": 1,
        "centerPointMicrons": {
            "x": coord[0],
            "y": coord[1]
        },
        "timingChoice": 7,
        "frameSizePixels": {
            "width": 2048,
            "height": 2048
        },
        "imagingPreset": {
            "preset": "Normal",
            "aperture": "2",
            "displayName": "Fine",
            "defaults": {
              "timingChoice": 7
            }
        },
        "sectionId": 8201,
        "slideId": 5931,
        "name": name,
        "timingDescription": "1 ms"
    }

    return sample_fov_tiling_entry


def generate_sample_fovs_list(fov_coords, fov_names):
    """Generate a sample dictionary of fovs for tiling

    Args:
        fov_coords (list):
            A list of tuples listing the starting x and y coordinates of each fov
        fov_names (list):
            A list of strings identifying the name of each fov

    Returns:
        dict:
            A dummy fovs list with starting x and y set to the provided coordinates and name
    """

    sample_fovs_list = {
        "exportDateTime": "2021-03-12T19:02:37.920Z",
        "fovFormatVersion": "1.5",
        "fovs": []
    }

    for coord, name in zip(fov_coords, fov_names):
        sample_fovs_list["fovs"].append(
            generate_sample_fov_tiling_entry(coord, name)
        )

    return sample_fovs_list
