import os
import warnings

import skimage.io as io
import numpy as np
import xarray as xr
from mibidata import tiff

from ark.utils import io_utils as iou


def load_imgs_from_mibitiff(data_dir, mibitiff_files=None, channels=None, delimiter=None,
                            dtype='int16'):
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
            name. Defaults to None
        dtype (str/type):
            optional specifier of image type.  Overwritten with warning for float images

    Returns:
        xarray.DataArray:
            xarray with shape [fovs, x_dim, y_dim, channels]
    """

    if not mibitiff_files:
        mibitiff_files = iou.list_files(data_dir, substrs=['.tif'])

    # extract fov names w/ delimiter agnosticism
    fovs = iou.extract_delimited_names(mibitiff_files, delimiter=delimiter)

    mibitiff_files = [os.path.join(data_dir, mt_file)
                      for mt_file in mibitiff_files]

    test_img = io.imread(mibitiff_files[0], plugin='tifffile')

    # check to make sure that float dtype was supplied if image data is float
    data_dtype = test_img.dtype
    if np.issubdtype(data_dtype, np.floating):
        if not np.issubdtype(dtype, np.floating):
            warnings.warn(f"The supplied non-float dtype {dtype} was overwritten to {data_dtype}, "
                          f"because the loaded images are floats")
            dtype = data_dtype

    # if no channels specified, get them from first MIBItiff file
    if channels is None:
        channel_tuples = tiff.read(mibitiff_files[0]).channels
        channels = [channel_tuple[1] for channel_tuple in channel_tuples]

    # extract images from MIBItiff file
    img_data = []
    for mibitiff_file in mibitiff_files:
        img_data.append(tiff.read(mibitiff_file)[channels])
    img_data = np.stack(img_data, axis=0)
    img_data = img_data.astype(dtype)

    # create xarray with image data
    img_xr = xr.DataArray(img_data,
                          coords=[fovs, range(img_data[0].data.shape[0]),
                                  range(img_data[0].data.shape[1]), channels],
                          dims=["fovs", "rows", "cols", "channels"])

    img_xr = img_xr.sortby('fovs').sortby('channels')

    return img_xr


def load_imgs_from_multitiff(data_dir, multitiff_files=None, channels=None, delimiter=None,
                             dtype='int16'):
    """Load images from a series of multi-channel tiff files.

    This function takes a set of multi-channel tiff files and loads the images into an xarray.
    The type used to store the images will be the same as that of the images stored in the
    multi-channel tiff files.

    This function differs from `load_imgs_from_mibitiff` in that proprietary metadata is unneeded,
    which is usefull loading in more general multi-channel tiff files.

    Args:
        data_dir (str):
            directory containing multitiffs
        multitiff_files (list):
            list of multi-channel tiff files to load.  If None, all multitiff files in data_dir
            are loaded.
        channels (list):
            optional list of channels to load.  Unlike MIBItiff, this must be given as a numeric
            list of indices, since there is no metadata containing channel names.
        delimiter (str):
            optional delimiter-character/string which separate fov names from the rest of the file
            name. Default is None.
        dtype (str/type):
            optional specifier of image type.  Overwritten with warning for float images

    Returns:
        xarray.DataArray:
            xarray with shape [fovs, x_dim, y_dim, channels]
    """

    if not multitiff_files:
        multitiff_files = iou.list_files(data_dir, substrs=['.tif'])

    # extract fov names w/ delimiter agnosticism
    fovs = iou.extract_delimited_names(multitiff_files, delimiter=delimiter)

    multitiff_files = [os.path.join(data_dir, mt_file)
                       for mt_file in multitiff_files]

    test_img = io.imread(multitiff_files[0], plugin='tifffile')

    # check to make sure that float dtype was supplied if image data is float
    data_dtype = test_img.dtype
    if np.issubdtype(data_dtype, np.floating):
        if not np.issubdtype(dtype, np.floating):
            warnings.warn(f"The supplied non-float dtype {dtype} was overwritten to {data_dtype}, "
                          f"because the loaded images are floats")
            dtype = data_dtype

    # extract data
    img_data = []
    for multitiff_file in multitiff_files:
        img_data.append(io.imread(multitiff_file, plugin='tifffile'))
    img_data = np.stack(img_data, axis=0)
    img_data = img_data.astype(dtype)

    if channels:
        img_data = img_data[:, :, :, channels]

    # create xarray with image data
    img_xr = xr.DataArray(img_data,
                          coords=[fovs, range(img_data.shape[1]),
                                  range(img_data.shape[2]),
                                  channels if channels else range(img_data.shape[3])],
                          dims=["fovs", "rows", "cols", "channels"])

    img_xr = img_xr.sortby('fovs').sortby('channels')

    return img_xr


def load_imgs_from_tree(data_dir, img_sub_folder=None, fovs=None, channels=None,
                        dtype="int16", variable_sizes=False):
    """Takes a set of imgs from a directory structure and loads them into an xarray.

    Args:
        data_dir (str):
            directory containing folders of images
        img_sub_folder (str):
            optional name of image sub-folder within each fov
        fovs (list):
            optional list of folders to load imgs from. Default loads all folders
        channels (list):
            optional list of imgs to load, otherwise loads all imgs
        dtype (str/type):
            dtype of array which will be used to store values
        variable_sizes (bool):
            if true, will pad loaded images with zeros to fit into array

    Returns:
        xarray.DataArray:
            xarray with shape [fovs, x_dim, y_dim, tifs]
    """

    if fovs is None:
        # get all fovs
        fovs = iou.list_folders(data_dir)
        fovs.sort()

    if len(fovs) == 0:
        raise ValueError(f"No fovs found in directory, {data_dir}")

    if img_sub_folder is None:
        # no img_sub_folder, change to empty string to read directly from base folder
        img_sub_folder = ""

    # get imgs from first fov if no img names supplied
    if channels is None:
        channels = iou.list_files(os.path.join(data_dir, fovs[0], img_sub_folder),
                                  substrs=['.tif', '.jpg', '.png'])

        # if taking all channels from directory, sort them alphabetically
        channels.sort()
    # otherwise, fill channel names with correct file extension
    elif not all([img.endswith(("tif", "tiff", "jpg", "png")) for img in channels]):
        channels = iou.list_files(
            os.path.join(data_dir, fovs[0], img_sub_folder),
            substrs=channels
        )

    if len(channels) == 0:
        raise ValueError("No images found in designated folder")

    test_img = io.imread(os.path.join(data_dir, fovs[0], img_sub_folder, channels[0]))

    # check to make sure that float dtype was supplied if image data is float
    data_dtype = test_img.dtype
    if np.issubdtype(data_dtype, np.floating):
        if not np.issubdtype(dtype, np.floating):
            warnings.warn(f"The supplied non-float dtype {dtype} was overwritten to {data_dtype}, "
                          f"because the loaded images are floats")
            dtype = data_dtype

    if variable_sizes:
        img_data = np.zeros((len(fovs), 1024, 1024, len(channels)), dtype=dtype)
    else:
        img_data = np.zeros((len(fovs), test_img.shape[0], test_img.shape[1], len(channels)),
                            dtype=dtype)

    for fov in range(len(fovs)):
        for img in range(len(channels)):
            if variable_sizes:
                temp_img = io.imread(
                    os.path.join(data_dir, fovs[fov], img_sub_folder, channels[img])
                )
                img_data[fov, :temp_img.shape[0], :temp_img.shape[1], img] = temp_img
            else:
                img_data[fov, :, :, img] = io.imread(os.path.join(data_dir, fovs[fov],
                                                                  img_sub_folder, channels[img]))

    # check to make sure that dtype wasn't too small for range of data
    if np.min(img_data) < 0:
        raise ValueError("Integer overflow from loading TIF image, try a larger dtype")

    if variable_sizes:
        row_coords, col_coords = range(1024), range(1024)
    else:
        row_coords, col_coords = range(test_img.shape[0]), range(test_img.shape[1])

    # remove .tif or .tiff from image name
    img_names = [os.path.splitext(img)[0] for img in channels]

    img_xr = xr.DataArray(img_data, coords=[fovs, row_coords, col_coords, img_names],
                          dims=["fovs", "rows", "cols", "channels"])

    # sort by fovs and channels for deterministic result
    img_xr = img_xr.sortby('fovs').sortby('channels')

    return img_xr


def load_imgs_from_dir(data_dir, files=None, delimiter=None, xr_dim_name='compartments',
                       xr_channel_names=None, dtype="int16", force_ints=False,
                       channel_indices=None):
    """Takes a set of images (possibly multitiffs) from a directory and loads them
     into an xarray.

    Args:
        data_dir (str):
            directory containing images
        files (list):
            list of files (e.g. ['fov1.tif'. 'fov2.tif'] to load.
            If None, all (.tif, .jpg, .png) files in data_dir are loaded.
        delimiter (str):
            character used to determine the file-prefix containging the fov name.
            Default is None.
        xr_dim_name (str):
            sets the name of the last dimension of the output xarray.
            Default: 'compartments'
        xr_channel_names (list):
            sets the name of the coordinates in the last dimension of the output xarray.
        dtype (str/type):
            data type to load/store
        force_ints (bool):
            If dtype is an integer, forcefully convert float imgs to ints. Default is False.
        channel_indices (list):
            optional list of imgs to load, if None or empty loads all imgs
            (relevant when loading multitiff images).

    Returns:
        xarray.DataArray:
            xarray with shape [fovs, x_dim, y_dim, tifs]

    Raises:
            ValueError:
                Raised if data_dir is not a directory, or if <data_dir>/img is
                not a file for some img in the input 'files' list.
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"Invalid value for data_dir. {data_dir} is not a directory.")

    if files is None:
        imgs = iou.list_files(data_dir, substrs=['.tif', '.jpg', '.png'])
    else:
        imgs = files
        for img in imgs:
            if not os.path.isfile(os.path.join(data_dir, img)):
                raise ValueError(f"Invalid value for {img}. "
                                 f"{os.path.join(data_dir, img)} is not a file.")

    if len(imgs) == 0:
        raise ValueError(f"No images found in directory, {data_dir}")

    test_img = io.imread(os.path.join(data_dir, imgs[0]))

    # check to make sure that float dtype was supplied if image data is float
    data_dtype = test_img.dtype
    if force_ints and np.issubdtype(dtype, np.integer):
        if not np.issubdtype(data_dtype, np.integer):
            warnings.warn(f"The the loaded {data_dtype} images were forcefully "
                          f"overwritten with the supplied integer dtype {dtype}")
    elif np.issubdtype(data_dtype, np.floating):
        if not np.issubdtype(dtype, np.floating):
            warnings.warn(f"The supplied non-float dtype {dtype} was overwritten to {data_dtype}, "
                          f"because the loaded images are floats")
            dtype = data_dtype

    # extract data
    img_data = []
    for img in imgs:
        v = io.imread(os.path.join(data_dir, img))
        if v.ndim == 2:
            v = v[:, :, np.newaxis]
        elif v.shape[0] == min(v.shape):
            # covert channels_first to be channels_last
            v = np.moveaxis(v, 0, -1)
        img_data.append(v)
    img_data = np.stack(img_data, axis=0)

    img_data = img_data.astype(dtype)

    if channel_indices:
        img_data = img_data[:, :, :, channel_indices]

    # check to make sure that dtype wasn't too small for range of data
    if np.min(img_data) < 0:
        raise ValueError("Integer overflow from loading TIF image, try a larger dtype")

    if test_img.ndim == 3 and test_img.shape[0] == min(test_img.shape):
        # channels_first
        row_coords, col_coords = range(test_img.shape[1]), range(test_img.shape[2])
    else:
        row_coords, col_coords = range(test_img.shape[0]), range(test_img.shape[1])

    # get fov name from imgs
    fovs = iou.extract_delimited_names(imgs, delimiter=delimiter)

    # create xarray with image data
    img_xr = xr.DataArray(img_data,
                          coords=[fovs, row_coords,
                                  col_coords,
                                  xr_channel_names if xr_channel_names else range(img_data.shape[3])],
                          dims=["fovs", "rows", "cols", xr_dim_name])

    img_xr = img_xr.sortby('fovs').sortby(xr_dim_name)

    return img_xr
