import os
from random import choices
from string import ascii_lowercase
import numpy as np
import pandas as pd
import xarray as xr
import skimage.io as io

from mibidata import mibi_image as mi, tiff


def gen_fov_chan_names(num_fovs, num_chans, return_imgs=False, use_delimiter=False):
    """ Generate FOV and channel names

    Names have the format 'Point0', 'Point1', ..., 'PointN' for FOVS and 'chan0', 'chan1', ...,
    'chanM' for channels.

    Args:
        num_fovs (int):
            Number of FOV names to create
        num_chans (int):
            Number of channel names to create
        return_imgs (bool):
            Return 'chanK.tiff' as well if True.  Default is False
        use_delimiter (bool):
            Appends '_otherinfo' to the first FOV.  Useful for testing FOV id extraction from
            filenames.  Default is False

    Returns:
        tuple (list, list) or (list, list, list):
            If return_imgs is False, only FOV and channel names are returned
            If return_imgs is True, image names will also be returned
    """
    fovs = [f'Point{i}' for i in range(num_fovs)]
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
    'fov_id': 'Point1', 'fov_name': 'R1C3_Tonsil',
    'folder': 'Point1/RowNumber0/Depth_Profile0',
    'dwell': 4, 'scans': '0,5', 'aperture': 'B',
    'instrument': 'MIBIscope1', 'tissue': 'Tonsil',
    'panel': '20170916_1x', 'mass_offset': 0.1, 'mass_gain': 0.2,
    'time_resolution': 0.5, 'miscalibrated': False, 'check_reg': False,
    'filename': '20180703_1234_test', 'description': 'test image',
    'version': 'alpha',
}


def _gen_tif_data(fov_number, chan_number, img_shape, fills, dtype):
    """ Generates random or set-filled image data

    Args:
        fov_number (int):
            Number of FOV's required
        chan_number (int):
            Number of channels required
        img_shape (tuple):
            Single image dimensions (x pixels, y pixels)
        fills (bool):
            If False, data is randomized.  If True, each single image will be filled with a value
            one less than that of the next channel.  If said image is the last channel, then the
            value is one less than that of the first channel in the next FOV.
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


def _create_tifs(base_dir, fov_names, img_names, shape, sub_dir, fills, dtype):
    tif_data = _gen_tif_data(len(fov_names), len(img_names), shape, fills, dtype)

    if sub_dir is None:
        sub_dir = ""

    filelocs = {}

    for i, fov in enumerate(fov_names):
        filelocs[fov] = []
        fov_path = os.path.join(base_dir, fov, sub_dir)
        os.makedirs(fov_path)
        for j, name in enumerate(img_names):
            io.imsave(os.path.join(fov_path, f'{name}.tiff'), tif_data[i, :, :, j])
            filelocs[fov].append(os.path.join(fov_path, name))

    return filelocs, tif_data


def _create_multitiff(base_dir, fov_names, channel_names, shape, sub_dir, fills, dtype):
    tif_data = _gen_tif_data(len(fov_names), len(channel_names), shape, fills, dtype)

    filelocs = {}

    for i, fov in enumerate(fov_names):
        tiffpath = os.path.join(base_dir, f'{fov}.tiff')
        io.imsave(tiffpath, tif_data[i, :, :, :], plugin='tifffile')
        filelocs[fov] = tiffpath

    return filelocs, tif_data


def _create_mibitiff(base_dir, fov_names, channel_names, shape, sub_dir, fills, dtype):
    tif_data = _gen_tif_data(len(fov_names), len(channel_names), shape, fills, dtype)

    filelocs = {}

    mass_map = tuple(enumerate(channel_names, 1))

    for i, fov in enumerate(fov_names):
        tif_obj = mi.MibiImage(tif_data[i, :, :, :],
                               mass_map,
                               **MIBITIFF_METADATA)

        tiffpath = os.path.join(base_dir, f'{fov}.tiff')
        tiff.write(tiffpath, tif_obj, dtype=dtype)
        filelocs[fov] = tiffpath

    return filelocs, tif_data


def _create_reverse_multitiff(base_dir, fov_names, channel_names, shape, sub_dir, fills, dtype):
    tif_data = _gen_tif_data(len(channel_names), len(fov_names), shape, fills, dtype)

    filelocs = {}

    for i, fov in enumerate(fov_names):
        tiffpath = os.path.join(base_dir, f'{fov}.tiff')
        io.imsave(tiffpath, tif_data[:, :, :, i], plugin='tifffile')
        filelocs[fov] = tiffpath

    tif_data = np.swapaxes(tif_data, 0, -1)

    return filelocs, tif_data


def _create_labels(base_dir, fov_names, comp_names, shape, sub_dir, fills, dtype):
    label_data = _gen_label_data(len(fov_names), len(comp_names), shape, dtype)

    filelocs = {}

    for i, fov in enumerate(fov_names):
        tiffpath = os.path.join(base_dir, f'{fov}.tiff')
        io.imsave(tiffpath, label_data[i, :, :, 0], plugin='tifffile')
        filelocs[fov] = tiffpath

    return filelocs, label_data


TIFFMAKERS = {
    'tiff': _create_tifs,
    'multitiff': _create_multitiff,
    'reverse_multitiff': _create_reverse_multitiff,
    'mibitiff': _create_mibitiff,
    'labels': _create_labels,
}


def create_paired_xarray_fovs(base_dir, fov_names, channel_names, img_shape=(10, 10),
                              mode='tiff', delimiter=None, sub_dir=None, fills=False,
                              dtype="int8"):

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

    filelocs, tif_data = TIFFMAKERS[mode](base_dir, fov_names, channel_names, img_shape, sub_dir,
                                          fills, dtype)

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
    if label_data is None:
        label_data = _gen_label_data(len(fov_ids), len(compartment_names), (row_size, col_size),
                                     dtype=dtype)
    else:
        row_size, col_size = label_data.shape[1:3]

        buf_fov_ids, _ = gen_fov_chan_names(label_data.shape[0], 0)
        if fov_ids is None:
            fov_ids = buf_fov_ids

    coords = [fov_ids, range(row_size), range(col_size), compartment_names]
    dims = ['fovs', 'rows', 'cols', 'compartments']
    return xr.DataArray(label_data, coords=coords, dims=dims)


TEST_MARKERS = list('ABCDEFG')


def make_segmented_csv(num_cells, extra_cols=None):
    cell_data = pd.DataFrame(
        np.random.random(size=(num_cells, len(TEST_MARKERS))),
        columns=TEST_MARKERS
    )
    cell_data["cell_type"] = choices(ascii_lowercase, k=num_cells)
    cell_data["PatientID"] = choices(range(1, 10), k=num_cells)

    return cell_data


def create_test_extraction_data():
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
