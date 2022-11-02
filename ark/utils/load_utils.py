import os
import warnings
import re

import numpy as np
import skimage.io as io
import xarray as xr
import natsort as ns

from ark.utils import io_utils as iou
from ark.utils import misc_utils
from ark.utils.tiff_utils import read_mibitiff


def load_imgs_from_mibitiff(data_dir, mibitiff_files=None, channels=None, delimiter=None):
    """Load images from a series of MIBItiff files.

    This function takes a set of MIBItiff files and load the images into an xarray. The type used
    to store the images will be the same as that of the MIBIimages stored in the MIBItiff files.

    Args:
        data_dir (str):
            directory containing MIBItiffs
        mibitiff_files (list):
            list of MIBItiff files to load. If None, all MIBItiff files in data_dir are loaded.
        channels (list):
            optional list of channels to load. Defaults to `None`, in which case, all channels in
            the first MIBItiff are used.
        delimiter (str):
            optional delimiter-character/string which separate fov names from the rest of the file
            name. Defaults to None.

    Returns:
        xarray.DataArray:
            xarray with shape [fovs, x_dim, y_dim, channels]
    """

    iou.validate_paths(data_dir, data_prefix=False)

    if not mibitiff_files:
        mibitiff_files = iou.list_files(data_dir, substrs=['.tiff'])
        mibitiff_files.sort()

    if len(mibitiff_files) == 0:
        raise ValueError("No mibitiff files specified in the data directory %s" % data_dir)

    # extract fov names w/ delimiter agnosticism
    fovs = iou.remove_file_extensions(mibitiff_files)
    fovs = iou.extract_delimited_names(fovs, delimiter=delimiter)

    mibitiff_files = [
        os.path.join(data_dir, mt_file)
        for mt_file in mibitiff_files
    ]

    test_img = io.imread(mibitiff_files[0], plugin='tifffile')

    # The dtype is always the type of the image being loaded in.
    dtype = test_img.dtype

    # if no channels specified, get them from first MIBItiff file
    if channels is None:
        _, channel_tuples = read_mibitiff(mibitiff_files[0])
        channels = [channel_tuple[1] for channel_tuple in channel_tuples]

    if len(channels) == 0:
        raise ValueError("No channels provided in channels list")

    # extract images from MIBItiff file
    img_data = []
    for mibitiff_file in mibitiff_files:
        img_data.append(read_mibitiff(mibitiff_file, channels)[0])
    img_data = np.stack(img_data, axis=0)

    if np.min(img_data) < 0:
        warnings.warn("You have images with negative values loaded in.")

    img_data = img_data.astype(dtype)

    # create xarray with image data
    img_xr = xr.DataArray(img_data,
                          coords=[fovs, range(img_data[0].data.shape[0]),
                                  range(img_data[0].data.shape[1]), channels],
                          dims=["fovs", "rows", "cols", "channels"])

    return img_xr


def load_imgs_from_tree(data_dir, img_sub_folder=None, fovs=None, channels=None,
                        max_image_size=None):
    """Takes a set of imgs from a directory structure and loads them into an xarray.

    Args:
        data_dir (str):
            directory containing folders of images
        img_sub_folder (str):
            optional name of image sub-folder within each fov
        fovs (str, list):
            optional list of folders to load imgs from, or the name of a single folder. Default
            loads all folders
        channels (list):
            optional list of imgs to load, otherwise loads all imgs
        max_image_size (int or None):
            The length (in pixels) of the largest image that will be loaded. All other images will
            be padded to bring them up to the same size.

    Returns:
        xarray.DataArray:
            xarray with shape [fovs, x_dim, y_dim, tifs]
    """

    iou.validate_paths(data_dir, data_prefix=False)

    if fovs is None:
        # get all fovs
        fovs = iou.list_folders(data_dir)
        fovs.sort()

    if len(fovs) == 0:
        raise ValueError(f"No fovs found in directory, {data_dir}")

    # If the fov provided is a single string (`fov_1` instead of [`fov_1`])
    if type(fovs) is str:
        fovs = [fovs]

    if img_sub_folder is None:
        # no img_sub_folder, change to empty string to read directly from base folder
        img_sub_folder = ""

    # get imgs from first fov if no img names supplied
    if channels is None:
        channels = iou.list_files(
            dir_name=os.path.join(data_dir, fovs[0], img_sub_folder),
            substrs=['.tiff', '.jpg', '.png']
        )

        # if taking all channels from directory, sort them alphabetically
        channels.sort()
    # otherwise, fill channel names with correct file extension
    elif not all([img.endswith(("tiff", "jpg", "png")) for img in channels]):
        # need this to reorder channels back because list_files may mess up the ordering
        channels_no_delim = [img.split('.')[0] for img in channels]

        all_channels = iou.list_files(
            dir_name=os.path.join(data_dir, fovs[0], img_sub_folder), substrs=channels_no_delim,
            exact_match=True
        )

        # get the corresponding indices found in channels_no_delim
        channels_indices = [channels_no_delim.index(chan.split('.')[0]) for chan in all_channels]

        # verify if channels from user input are present in `all_channels`
        all_channels_no_delim = [channel.split('.')[0] for channel in all_channels]

        misc_utils.verify_same_elements(all_channels_in_folder=all_channels_no_delim,
                                        all_channels_detected=channels_no_delim)
        # reorder back to original
        channels = [chan for _, chan in sorted(zip(channels_indices, all_channels))]

    if len(channels) == 0:
        raise ValueError("No images found in designated folder")

    test_img = io.imread(
        os.path.join(data_dir, fovs[0], img_sub_folder, channels[0])
    )

    # The dtype is always the type of the image being loaded in.
    dtype = test_img.dtype

    if max_image_size is not None:
        img_data = np.zeros((len(fovs), max_image_size, max_image_size, len(channels)),
                            dtype=dtype)
    else:
        img_data = np.zeros((len(fovs), test_img.shape[0], test_img.shape[1], len(channels)),
                            dtype=dtype)

    for fov in range(len(fovs)):
        for img in range(len(channels)):
            if max_image_size is not None:
                temp_img = io.imread(
                    os.path.join(data_dir, fovs[fov], img_sub_folder, channels[img])
                )
                img_data[fov, :temp_img.shape[0], :temp_img.shape[1], img] = temp_img
            else:
                img_data[fov, :, :, img] = io.imread(
                    os.path.join(data_dir, fovs[fov], img_sub_folder, channels[img])
                )

    # check to make sure that dtype wasn't too small for range of data
    if np.min(img_data) < 0:
        warnings.warn("You have images with negative values loaded in.")

    row_coords, col_coords = range(img_data.shape[1]), range(img_data.shape[2])

    # remove .tiff from image name
    img_names = [os.path.splitext(img)[0] for img in channels]

    img_xr = xr.DataArray(img_data, coords=[fovs, row_coords, col_coords, img_names],
                          dims=["fovs", "rows", "cols", "channels"])

    return img_xr


def load_imgs_from_dir(data_dir, files=None, match_substring=None, trim_suffix=None,
                       xr_dim_name='compartments', xr_channel_names=None, channel_indices=None):
    """Takes a set of images (possibly multitiffs) from a directory and loads them into an xarray.

    Args:
        data_dir (str):
            directory containing images
        files (list):
            list of files (e.g. ['fov1.tiff'. 'fov2.tiff'] to load.
            If None, all (.tiff, .jpg, .png) files in data_dir are loaded.
        match_substring (str):
            a filename substring that all loaded images must contain. Ignored if files argument is
            not None.  If None, no matching is performed.
            Default is None.
        trim_suffix (str):
            a filename suffix to trim from the fov name if present. If None, no characters will be
            trimmed.  Default is None.
        xr_dim_name (str):
            sets the name of the last dimension of the output xarray.
            Default: 'compartments'
        xr_channel_names (list):
            sets the name of the coordinates in the last dimension of the output xarray.
        channel_indices (list):
            optional list of indices specifying which channels to load (by their indices).
            if None or empty, the function loads all channels.
            (Ignored if data is not multitiff).

    Returns:
        xarray.DataArray:
            xarray with shape [fovs, x_dim, y_dim, tifs]

    Raises:
        ValueError:
            Raised in the following cases:

            - data_dir is not a directory, <data_dir>/img is
              not a file for some img in the input 'files' list, or no images are found.
            - channels_indices are invalid according to the shape of the images.
            - The length of xr_channel_names (if provided) does not match the number
              of channels in the input.
    """

    iou.validate_paths(data_dir, data_prefix=False)

    if files is None:
        imgs = iou.list_files(data_dir, substrs=['.tiff', '.jpg', '.png'])
        if match_substring is not None:
            filenames = iou.remove_file_extensions(imgs)
            imgs = [imgs[i] for i, name in enumerate(filenames) if match_substring in name]
        imgs.sort()
    else:
        imgs = files
        for img in imgs:
            if not os.path.isfile(os.path.join(data_dir, img)):
                raise ValueError(f"Invalid value for {img}. "
                                 f"{os.path.join(data_dir, img)} is not a file.")

    if len(imgs) == 0:
        raise ValueError(f"No images found in directory, {data_dir}")

    test_img = io.imread(os.path.join(data_dir, imgs[0]))

    # check data format
    multitiff = test_img.ndim == 3
    channels_first = multitiff and test_img.shape[0] == min(test_img.shape)

    # check to make sure all channel indices are valid given the shape of the image
    n_channels = 1
    if multitiff:
        n_channels = test_img.shape[0] if channels_first else test_img.shape[2]
        if channel_indices:
            if max(channel_indices) >= n_channels or min(channel_indices) < 0:
                raise ValueError(f'Invalid value for channel_indices. Indices should be'
                                 f' between 0-{n_channels-1} for the given data.')
    # make sure channels_names has the same length as the number of channels in the image
    if xr_channel_names and n_channels != len(xr_channel_names):
        raise ValueError(f'Invalid value for xr_channel_names. xr_channel_names'
                         f' length should be {n_channels}, as the number of channels'
                         f' in the input data.')

    # The dtype is always the type of the image being loaded in.
    dtype = test_img.dtype

    # extract data
    img_data = []
    for img in imgs:
        v = io.imread(os.path.join(data_dir, img))
        if not multitiff:
            v = np.expand_dims(v, axis=2)
        elif channels_first:
            # covert channels_first to be channels_last
            v = np.moveaxis(v, 0, -1)
        img_data.append(v)
    img_data = np.stack(img_data, axis=0)

    img_data = img_data.astype(dtype)

    if channel_indices and multitiff:
        img_data = img_data[:, :, :, channel_indices]

    if np.min(img_data) < 0:
        warnings.warn("You have images with negative values loaded in.")

    if channels_first:
        row_coords, col_coords = range(test_img.shape[1]), range(test_img.shape[2])
    else:
        row_coords, col_coords = range(test_img.shape[0]), range(test_img.shape[1])

    # get fov name from imgs
    fovs = iou.remove_file_extensions(imgs)
    fovs = iou.extract_delimited_names(fovs, delimiter=trim_suffix)

    # create xarray with image data
    img_xr = xr.DataArray(img_data,
                          coords=[fovs, row_coords, col_coords,
                                  xr_channel_names if xr_channel_names
                                  else range(img_data.shape[3])],
                          dims=["fovs", "rows", "cols", xr_dim_name])

    return img_xr


def check_fov_name_prefix(fov_list):
    """Checks for a prefix (usually detailing a run name) in any of the provided FOV names

        Args:
            fov_list (list): list of fov name
        Returns:
            tuple: (bool) whether at least one fov names has a prefix,
                   (list / dict) if prefix, dictionary with fov name as keys and prefixes as values
                    otherwise return a simple list of the fov names
        """

    # check for prefix in any of the fov names
    prefix = False
    for folder in fov_list:
        if re.search('R.{1,3}C.{1,3}', folder).start() != 0:
            prefix = True

    if prefix:
        # dict containing fov name and run name
        fov_names = {}
        for folder in fov_list:
            fov = ''.join(folder.split("_")[-1:])
            prefix_name = '_'.join(folder.split("_")[:-1])
            fov_names[fov] = prefix_name
    else:
        # original list of fov names
        fov_names = fov_list

    return prefix, fov_names


def get_tiled_fov_names(fov_list, return_dims=False):
    """Generates the complete tiled fov list when given a list of fov names

    Args:
        fov_list (list):
            list of fov names
        return_dims (bool):
            whether to also return row and col dimensions
    Returns:
        tuple: names of all fovs expected for tiled image shape, and dimensions if return_dims
    """

    rows, cols, expected_fovs = [], [], []

    # check for run name prefix
    prefix, fov_names = check_fov_name_prefix(fov_list)

    # get tiled image dimensions
    for fov in fov_names:
        fov_digits = re.findall(r'\d+', fov)
        rows.append(int(fov_digits[0]))
        cols.append(int(fov_digits[1]))
    row_num, col_num = max(rows), max(cols)

    # fill list of expected fov names
    for n in range(row_num):
        for m in range(col_num):
            fov = f'R{n + 1}C{m + 1}'
            # prepend run names
            if prefix and fov in list(fov_names.keys()):
                expected_fovs.append(f"{fov_names[fov]}_" + fov)
            else:
                expected_fovs.append(fov)

    if return_dims:
        return expected_fovs, row_num, col_num
    else:
        return expected_fovs


def load_tiled_img_data(data_dir, fovs, expected_fovs, channel, single_dir, file_ext='tiff',
                        img_sub_folder=''):
    """Takes a set of images from a directory structure and loads them into a tiled xarray.

    Args:
        data_dir (str):
            directory containing folders of images
        fovs (list/dict):
            list of fovs (or dictionary with folder and RnCm names) to load data for
        expected_fovs (list):
            list of all expected RnCm fovs names in the tiled grid
        channel (str):
            single image name to load
        single_dir (bool):
            whether the images are stored in a single directory rather than within fov subdirs
        file_ext (str):
            the file type of existing images
        img_sub_folder (str):
            optional name of image sub-folder within each fov

    Returns:
        xarray.DataArray:
            xarray with shape [fovs, x_dim, y_dim, channel]
    """

    iou.validate_paths(data_dir, data_prefix=False)

    # check for toffy fovs
    if type(fovs) is dict:
        fov_list = list(fovs.values())
        tiled_names = list(fovs.keys())
    else:
        fov_list = fovs
        tiled_names = []

    # no missing fov images, load data normally and return array
    if len(fov_list) == len(expected_fovs):
        if single_dir:
            img_xr = load_imgs_from_dir(data_dir, match_substring=channel, xr_dim_name='channels',
                                        trim_suffix='_' + channel, xr_channel_names=[channel])
        else:
            img_xr = load_imgs_from_tree(data_dir, img_sub_folder, fovs=fov_list,
                                         channels=[channel])
        return img_xr

    # missing fov directories, read in a test image to get data type
    if single_dir:
        test_path = os.path.join(data_dir, expected_fovs[0] + '_' + channel + '.' + file_ext)
    else:
        test_path = os.path.join(os.path.join(data_dir, fov_list[0], img_sub_folder, channel + '.'
                                              + file_ext))
    test_img = io.imread(test_path)
    img_data = np.zeros((len(expected_fovs), test_img.shape[0], test_img.shape[1], 1),
                        dtype=test_img.dtype)

    for fov, fov_name in enumerate(expected_fovs):
        # load in fov data for images, leave missing fovs as zeros
        if fov_name in fov_list:
            if single_dir:
                temp_img = io.imread(os.path.join(data_dir, fov_name + '_' +
                                                  channel + '.' + file_ext))
            else:
                temp_img = io.imread(os.path.join(data_dir, fov_name, img_sub_folder,
                                                  channel + '.' + file_ext))
            # fill in specific spot in array
            img_data[fov, :temp_img.shape[0], :temp_img.shape[1], 0] = temp_img

        # check against tiled_names from dict for toffy dirs
        elif fov_name in tiled_names:
            folder_name = fovs[fov_name]
            temp_img = io.imread(os.path.join(data_dir, folder_name, img_sub_folder,
                                              channel + '.' + file_ext))
            # fill in specific spot in array
            img_data[fov, :temp_img.shape[0], :temp_img.shape[1], 0] = temp_img

    # check to make sure that dtype wasn't too small for range of data
    if np.min(img_data) < 0:
        warnings.warn("You have images with negative values loaded in.")

    row_coords, col_coords = range(img_data.shape[1]), range(img_data.shape[2])

    img_xr = xr.DataArray(img_data, coords=[expected_fovs, row_coords, col_coords, [channel]],
                          dims=["fovs", "rows", "cols", "channels"])

    return img_xr
