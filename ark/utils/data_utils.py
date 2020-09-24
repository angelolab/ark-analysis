import os
import math
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


def load_imgs_from_dir(data_dir, imgdim_name='compartments', image_name='img_data', delimiter=None,
                       dtype="int16", variable_sizes=False, force_ints=False):
    """Takes a set of images from a directory and loads them into an xarray based on filename
    prefixes.

    Args:
        data_dir (str):
            directory containing images
        imgdim_name (str):
            sets the name of the last dimension of the output xarray
        image_name (str):
            sets name of the last coordinate in the output xarray
        delimiter (str):
            character used to determine the file-prefix containging the fov name. Default is None.
        dtype (str/type):
            data type to load/store
        variable_sizes (bool):
            Dynamically determine image sizes and pad smaller imgs w/ zeros
        force_ints (bool):
            If dtype is an integer, forcefully convert float imgs to ints. Default is False.

    Returns:
        xarray.DataArray:
            xarray with shape [fovs, x_dim, y_dim, 1]

    """

    imgs = iou.list_files(data_dir, substrs=['.tif', '.jpg', '.png'])

    imgs.sort()

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

    if variable_sizes:
        img_data = np.zeros((len(imgs), 1024, 1024, 1), dtype=dtype)
    else:
        img_data = np.zeros((len(imgs), test_img.shape[0], test_img.shape[1], 1),
                            dtype=dtype)

    for img in range(len(imgs)):
        if variable_sizes:
            temp_img = io.imread(os.path.join(data_dir, imgs[img]))
            img_data[img, :temp_img.shape[0], :temp_img.shape[1], 0] = temp_img.astype(dtype)
        else:
            img_data[img, :, :, 0] = io.imread(os.path.join(data_dir, imgs[img])).astype(dtype)

    # check to make sure that dtype wasn't too small for range of data
    if np.min(img_data) < 0:
        raise ValueError("Integer overflow from loading TIF image, try a larger dtype")

    if variable_sizes:
        row_coords, col_coords = range(1024), range(1024)
    else:
        row_coords, col_coords = range(test_img.shape[0]), range(test_img.shape[1])

    # get fov name from imgs
    fovs = iou.extract_delimited_names(imgs, delimiter=delimiter)

    img_xr = xr.DataArray(img_data.astype(dtype),
                          coords=[fovs, row_coords, col_coords, [image_name]],
                          dims=["fovs", "rows", "cols",
                          imgdim_name])

    # sort for deterministic fov names
    img_xr = img_xr.sortby('fovs')

    return img_xr


# TODO: Add metadata for channel name (eliminates need for fixed-order channels)
def generate_deepcell_input(data_xr, data_dir, nuc_channels, mem_channels):
    """Saves nuclear and membrane channels into deepcell input format.

    Writes summed channel images out as multitiffs

    Args:
        data_xr (xr.DataArray):
            xarray containing nuclear and membrane channels over many fov's
        data_dir (str):
            location to save deepcell input tifs
        nuc_channels (list):
            nuclear channels to be summed over
        mem_channels (list):
            membrane channels to be summed over
    """
    for fov in data_xr.fovs.values:
        out = np.zeros((data_xr.shape[1], data_xr.shape[2], 2), dtype=data_xr.dtype)

        # sum over channels and add to output
        if nuc_channels:
            out[:, :, 0] = \
                np.sum(data_xr.loc[fov, :, :, nuc_channels].values,
                       axis=2)
        if mem_channels:
            out[:, :, 1] = \
                np.sum(data_xr.loc[fov, :, :, mem_channels].values,
                       axis=2)

        save_path = os.path.join(data_dir, f'{fov}.tif')
        io.imsave(save_path, out, plugin='tifffile')


def combine_xarrays(xarrays, axis):
    """Combines a number of xarrays together

    Args:
        xarrays (tuple):
            a tuple of xarrays
        axis (int):
            either 0, if the xarrays will combined over different fovs, or -1 if they will be
            combined over channels

    Returns:
        xarray.DataArray:
            an xarray that is the combination of all inputs
    """

    first_xr = xarrays[0]
    np_arr = first_xr.values

    # define iterator to hold coord values of dimension that is being stacked
    if axis == 0:
        iterator = first_xr.fovs.values
        shape_slice = slice(1, 4)
    else:
        iterator = first_xr.channels.values
        shape_slice = slice(0, 3)

    # loop through each xarray, stack the coords, and concatenate the values
    for cur_xr in xarrays[1:]:
        cur_arr = cur_xr.values

        if cur_arr.shape[shape_slice] != first_xr.shape[shape_slice]:
            raise ValueError("xarrays have conflicting sizes")

        if axis == 0:
            if not np.array_equal(cur_xr.channels, first_xr.channels):
                raise ValueError("xarrays have different channels")
        else:
            if not np.array_equal(cur_xr.fovs, first_xr.fovs):
                raise ValueError("xarrays have different fovs")

        np_arr = np.concatenate((np_arr, cur_arr), axis=axis)
        if axis == 0:
            iterator = np.append(iterator, cur_xr.fovs.values)
        else:
            iterator = np.append(iterator, cur_xr.channels.values)

    # assign iterator to appropriate coord label
    if axis == 0:
        fovs = iterator
        channels = first_xr.channels.values
    else:
        fovs = first_xr.fovs.values
        channels = iterator

    combined_xr = xr.DataArray(np_arr, coords=[fovs, range(first_xr.shape[1]),
                                               range(first_xr.shape[2]), channels],
                               dims=["fovs", "rows", "cols", "channels"])

    return combined_xr


def crop_helper(image_stack, crop_size):
    """"Helper function to take an image, and return crops of size crop_size

    Args:
        image_stack (np.array):
            A 4D numpy array of shape(points, rows, columns, channels)
        crop_size (int):
            Size of the crop to take from the image. Assumes square crops

    Returns:
        numpy.ndarray:
            A 4D numpy array of shape (crops, rows, columns, channels)
    """

    if len(image_stack.shape) != 4:
        raise ValueError("Incorrect dimensions of input image. "
                         "Expecting 3D, got {}".format(image_stack.shape))

    # figure out number of crops for final image
    crop_num_row = math.ceil(image_stack.shape[1] / crop_size)
    crop_num_col = math.ceil(image_stack.shape[2] / crop_size)
    cropped_images = np.zeros(
        (crop_num_row * crop_num_col * image_stack.shape[0],
            crop_size, crop_size, image_stack.shape[3]),
        dtype=image_stack.dtype)

    # Determine if image will need to be padded with zeros due to uneven division by crop
    if image_stack.shape[1] % crop_size != 0 or image_stack.shape[2] % crop_size != 0:
        # create new array that is padded by one crop size on image dimensions
        new_shape = (image_stack.shape[0], image_stack.shape[1] + crop_size,
                     image_stack.shape[2] + crop_size, image_stack.shape[3])
        new_stack = np.zeros(new_shape, dtype=image_stack.dtype)
        new_stack[:, :image_stack.shape[1], :image_stack.shape[2], :] = image_stack
        image_stack = new_stack

    # iterate through the image row by row, cropping based on identified threshold
    img_idx = 0
    for point in range(image_stack.shape[0]):
        for row in range(crop_num_row):
            for col in range(crop_num_col):
                cropped_images[img_idx, :, :, :] = \
                    image_stack[point, (row * crop_size):((row + 1) * crop_size),
                                (col * crop_size):((col + 1) * crop_size), :]
                img_idx += 1

    return cropped_images


def crop_image_stack(image_stack, crop_size, stride_fraction):
    """Function to generate a series of tiled crops across an image.

    The tiled crops can overlap each other, with the overlap between tiles determined by the
    stride fraction. A stride fraction of 0.333 will move the window over 1/3 of the crop_size in
    x and y at each step, whereas a stride fraction of 1 will move the window the entire crop size
    at each iteration.

    Args:
        image_stack (np.array):
            A 4D numpy array of shape(points, rows, columns, channels)
        crop_size (int):
            size of the crop to take from the image. Assumes square crops
        stride_fraction (float):
            the relative size of the stride for overlapping crops as a function of the crop size.
    Returns:
        numpy.ndarray:
            A 4D numpy array of shape(crops, rows, cols, channels)
    """

    if len(image_stack.shape) != 4:
        raise ValueError("Incorrect dimensions of input image. "
                         "Expecting 3D, got {}".format(image_stack.shape))

    if crop_size > image_stack.shape[1]:
        raise ValueError("Invalid crop size: img shape is {} "
                         "and crop size is {}".format(image_stack.shape, crop_size))

    if stride_fraction > 1:
        raise ValueError("Invalid stride fraction. Must be less than 1, "
                         "passed a value of {}".format(stride_fraction))

    # Determine how many distinct grids will be generated across the image
    stride_step = math.floor(crop_size * stride_fraction)
    num_strides = math.floor(1 / stride_fraction)

    for row_shift in range(num_strides):
        for col_shift in range(num_strides):

            if row_shift == 0 and col_shift == 0:
                # declare data holder
                cropped_images = crop_helper(image_stack, crop_size)
            else:
                # crop the image by the shift prior to generating grid of crops
                img_shift = image_stack[:, (row_shift * stride_step):,
                                        (col_shift * stride_step):, :]
                # print("shape of the input image is {}".format(img_shift.shape))
                temp_images = crop_helper(img_shift, crop_size)
                cropped_images = np.append(cropped_images, temp_images, axis=0)

    return cropped_images


def combine_point_directories(dir_path):
    """Combines a folder containing multiple imaging runs into a single folder

    Args:
        dir_path (str):
            path to directory containing the sub directories
    """

    if not os.path.exists(dir_path):
        raise ValueError("Directory does not exist")

    # gets all sub folders
    folders = os.listdir(dir_path)
    folders = [folder for folder in folders if os.path.isdir(os.path.join(dir_path, folder))]

    os.makedirs(os.path.join(dir_path, "combined_folder"))

    # loop through sub folders, get all contents, and transfer to new folder
    for folder in folders:
        points = os.listdir(os.path.join(dir_path, folder))
        print(points)
        for point in points:
            os.rename(os.path.join(dir_path, folder, point),
                      os.path.join(dir_path, "combined_folder", folder + "_" + point))


def stitch_images(data_xr, num_cols):
    """Stitch together a stack of different channels from different FOVs into a single 2D image
    for each channel

    Args:
        data_xr (xarray.DataArray):
            xarray containing image data from multiple fovs and channels
        num_cols (int):
            number of images stitched together horizontally

    Returns:
        xarray.DataArray:
            the stitched image data
    """
    num_imgs = data_xr.shape[0]
    num_rows = math.ceil(num_imgs / num_cols)
    row_len = data_xr.shape[1]
    col_len = data_xr.shape[2]

    total_row_len = num_rows * row_len
    total_col_len = num_cols * col_len

    stitched_data = np.zeros((1, total_row_len, total_col_len, data_xr.shape[3]),
                             dtype=data_xr.dtype)

    img_idx = 0
    for row in range(num_rows):
        for col in range(num_cols):
            stitched_data[0, row * row_len:(row + 1) * row_len,
                          col * col_len:(col + 1) * col_len, :] = data_xr[img_idx, ...]
            img_idx += 1
            if img_idx == num_imgs:
                break

    stitched_xr = xr.DataArray(stitched_data, coords=[['stitched_image'], range(total_row_len),
                                                      range(total_col_len), data_xr.channels],
                               dims=['points', 'rows', 'cols', 'channels'])
    return stitched_xr


def split_img_stack(stack_dir, output_dir, stack_list, indices, names, channels_first=True):
    """Splits the channels in a given directory of images into separate files

    Images are saved in the output_dir

    Args:
        stack_dir (str):
            where we read the input files
        output_dir (str):
            where we write the split channel data
        stack_list (list):
            the names of the files we want to read from stack_dir
        indices (list):
            the indices we want to pull data from
        names (list):
            the corresponding names of the channels
        channels_first (bool):
            whether we index at the beginning or end of the array
    """

    for stack_name in stack_list:
        img_stack = io.imread(os.path.join(stack_dir, stack_name))
        img_dir = os.path.join(output_dir, os.path.splitext(stack_name)[0])
        os.makedirs(img_dir)

        for i in range(len(indices)):
            if channels_first:
                channel = img_stack[indices[i], ...]
            else:
                channel = img_stack[..., indices[i]]
            io.imsave(os.path.join(os.path.join(img_dir, names[i])), channel)
