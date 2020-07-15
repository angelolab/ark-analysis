import os
import pathlib
import math

import skimage.io as io
import numpy as np
import xarray as xr
from mibidata import tiff


def validate_paths(paths):
    """Verifys that paths exist and don't leave Docker's scope

    Args:
        paths: paths to verify.

    Output:
        Raises errors if any directory is out of scope or non-existent
    """

    # if given a single path, convert to list
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        if not os.path.exists(path):
            if str(path).startswith('../data'):
                for parent in reversed(pathlib.Path(path).parents):
                    if not os.path.exists(parent):
                        raise ValueError(
                            f'A bad path, {path}, was provided.\n'
                            f'The folder, {parent.name}, could not be found...')
                raise ValueError(
                    f'The file/path, {pathlib.Path(path).name}, could not be found...')
            else:
                raise ValueError(
                    f'The path, {path}, is not prefixed with \'../data\'.\n'
                    f'Be sure to add all images/files/data to the \'data\' folder, '
                    f'and to reference as \'../data/path_to_data/myfile.tif\'')


def load_imgs_from_mibitiff(mibitiff_paths, channels=None):
    """Load images from a series of MIBItiff files.

    This function takes a set of MIBItiff files and load the images into an
    xarray. The type used to store the images will be the same as that of the
    MIBIimages stored in the MIBItiff files.

    Args:
        mibitiff_paths: list of MIBItiff files to load.
        channels: optional list of channels to load. Defaults to `None`, in
            which case, all channels in the first MIBItiff are used.

    Returns:
        img_xr: xarray with shape [fovs, tifs, x_dim, y_dim]
    """

    validate_paths(mibitiff_paths)

    # if no channels specified, get them from first MIBItiff file
    if channels is None:
        channel_tuples = tiff.read(mibitiff_paths[0]).channels
        channels = [channel_tuple[1] for channel_tuple in channel_tuples]

    # extract point name from file name
    fovs = [mibitiff_path.split(os.sep)[-1].split('_')[0] for mibitiff_path
            in mibitiff_paths]

    # extract images from MIBItiff file
    img_data = []
    for mibitiff_path in mibitiff_paths:
        img_data.append(tiff.read(mibitiff_path)[channels])
    img_data = np.stack(img_data, axis=0)

    # create xarray with image data
    img_xr = xr.DataArray(img_data,
                          coords=[fovs, range(img_data[0].data.shape[0]),
                                  range(img_data[0].data.shape[1]), channels],
                          dims=["fovs", "rows", "cols", "channels"])

    return img_xr


def load_imgs_from_tree(data_dir, img_sub_folder=None, fovs=None, imgs=None,
                        dtype="int16", variable_sizes=False):
    """Takes a set of imgs from a directory structure and loads them into an xarray.

        Args:
            data_dir: directory containing folders of images
            img_sub_folder: optional name of image sub-folder within each fov
            fovs: optional list of folders to load imgs from, otherwise loads from all folders
            imgs: optional list of imgs to load, otherwise loads all imgs
            dtype: dtype of array which will be used to store values
            variable_sizes: if true, will pad loaded images with zeros to fit into array

        Returns:
            img_xr: xarray with shape [fovs, x_dim, y_dim, tifs]
    """

    validate_paths(data_dir)

    if fovs is None:
        # get all fovs
        fovs = os.listdir(data_dir)
        fovs = [fov for fov in fovs if os.path.isdir(os.path.join(data_dir, fov))]
        fovs.sort()
    else:
        # use supplied list, but check to make sure they all exist
        validate_paths([os.path.join(data_dir, fov) for fov in fovs])

    if len(fovs) == 0:
        raise ValueError("No fovs found in directory")

    # check to make sure img subfolder name within fov is correct
    if img_sub_folder is not None:
        validate_paths(os.path.join(data_dir, fovs[0], img_sub_folder))
    else:
        # no img_sub_folder, change to empty string to read directly from base folder
        img_sub_folder = ""

    # get imgs from first fov if no img names supplied
    if imgs is None:
        imgs = os.listdir(os.path.join(data_dir, fovs[0], img_sub_folder))
        imgs = [img for img in imgs if np.isin(img.split(".")[-1], ["tif", "tiff", "jpg", "png"])]

        # if taking all imgs from directory, sort them alphabetically
        imgs.sort()

    if len(imgs) == 0:
        raise ValueError("No imgs found in designated folder")

    # check to make sure supplied imgs exist
    for img in imgs:
        if not os.path.isfile(os.path.join(data_dir, fovs[0], img_sub_folder, img)):
            raise ValueError("Could not find {} in supplied directory {}".format(
                img, os.path.join(data_dir, fovs[0], img_sub_folder, img)))

    test_img = io.imread(os.path.join(data_dir, fovs[0], img_sub_folder, imgs[0]))

    # check to make sure that float dtype was supplied if image data is float
    data_dtype = test_img.dtype
    if np.issubdtype(data_dtype, np.floating):
        if not np.issubdtype(dtype, np.floating):
            raise ValueError("supplied dtype is not a float, but the images loaded are floats")

    if variable_sizes:
        img_data = np.zeros((len(fovs), 1024, 1024, len(imgs)), dtype=dtype)
    else:
        img_data = np.zeros((len(fovs), test_img.shape[0], test_img.shape[1], len(imgs)),
                            dtype=dtype)

    for fov in range(len(fovs)):
        for img in range(len(imgs)):
            if variable_sizes:
                temp_img = io.imread(os.path.join(data_dir, fovs[fov], img_sub_folder, imgs[img]))
                img_data[fov, :temp_img.shape[0], :temp_img.shape[1], img] = temp_img
            else:
                img_data[fov, :, :, img] = io.imread(os.path.join(data_dir, fovs[fov],
                                                                  img_sub_folder, imgs[img]))

    # check to make sure that dtype wasn't too small for range of data
    if np.min(img_data) < 0:
        raise ValueError("Integer overflow from loading TIF image, try a larger dtype")

    if variable_sizes:
        row_coords, col_coords = range(1024), range(1024)
    else:
        row_coords, col_coords = range(test_img.shape[0]), range(test_img.shape[1])

    # remove .tif or .tiff from image name
    img_names = [os.path.splitext(img)[0] for img in imgs]

    img_xr = xr.DataArray(img_data, coords=[fovs, row_coords, col_coords, img_names],
                          dims=["fovs", "rows", "cols", "channels"])

    return img_xr


def load_imgs_from_dir(data_dir, image_name='img_data', delimiter='_',
                       dtype="int16", variable_sizes=False):
    """Takes a set of images from a directory and loads them into an xarray based on filename
    prefixes.

        Args:
            data_dir: directory containing images
            image_name: sets name of the 'channel' coordinate in the output xarray
            delimiter: character used to determine the file-prefix containging the fov name

        Returns:
            img_xr: xarray with shape [fovs, x_dim, y_dim, 1]

    """
    validate_paths(data_dir)

    imgs = os.listdir(data_dir)
    imgs = [img for img in imgs if np.isin(img.split(".")[-1], ["tif", "tiff", "jpg", "png"])]
    imgs.sort()

    if len(imgs) == 0:
        raise ValueError(f"No images found in directory, {data_dir}")

    test_img = io.imread(os.path.join(data_dir, imgs[0]))

    # check to make sure that float dtype was supplied if image data is float
    data_dtype = test_img.dtype
    if np.issubdtype(data_dtype, np.floating):
        if not np.issubdtype(dtype, np.floating):
            raise ValueError("supplied dtype is not a float, but the images loaded are floats")

    if variable_sizes:
        img_data = np.zeros((len(imgs), 1024, 1024, 1), dtype=dtype)
    else:
        img_data = np.zeros((len(imgs), test_img.shape[0], test_img.shape[1], 1),
                            dtype=dtype)

    for img in range(len(imgs)):
        if variable_sizes:
            temp_img = io.imread(os.path.join(data_dir, imgs[img]))
            img_data[img, :temp_img.shape[0], :temp_img.shape[1], 0] = temp_img
        else:
            img_data[img, :, :, 0] = io.imread(os.path.join(data_dir, imgs[img]))

    # check to make sure that dtype wasn't too small for range of data
    if np.min(img_data) < 0:
        raise ValueError("Integer overflow from loading TIF image, try a larger dtype")

    if variable_sizes:
        row_coords, col_coords = range(1024), range(1024)
    else:
        row_coords, col_coords = range(test_img.shape[0]), range(test_img.shape[1])

    # get fov name from imgs
    fovs = [img.split(delimiter)[0] for img in imgs]

    img_xr = xr.DataArray(img_data, coords=[fovs, row_coords, col_coords, [image_name]],
                          dims=["fovs", "rows", "cols", "channels"])

    return img_xr


# TODO: add test function
def tiffs_to_xr_labels(tiff_dir, output_dir=None, delimiter='_'):
    """Converts and condenses segmentation labels from a folder of tiff files
    into an xarray.  Tiff files are assumed to be prefixed with their respective fov
    names:

    Point1234_moreinfo.tiff

    Inputs:
        tiff_dir: path to directory containing segmentation labels as TIFs
        output_dir: path to directory where the output will be saved. If None, no xr is saved
        delimiter: character which separates fov-id prefix from the rest of the filename
    Outputs:
        Returns xarray and optionally saves it to output directory
    """
    im_xr = load_imgs_from_dir(data_dir=tiff_dir, image_name='segmentation_label',
                               delimiter=delimiter)

    if output_dir is not None:
        save_name = os.path.join(output_dir, 'segmentation_labels.xr')
        if os.path.exists(save_name):
            print("overwriting previously generated processed output file")
            os.remove(save_name)

        im_xr.to_netcdf(save_name, format="NETCDF3_64BIT")

    return im_xr


def combine_xarrays(xarrays, axis):
    """Combines a number of xarrays together

    Inputs:
        xarrays: a tuple of xarrays
        axis: either 0, if the xarrays will combined over different fovs,
        or -1 if they will be combined over channels

    Outputs:
        combined_xr: an xarray that is the combination of all inputs"""

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

    Inputs:
        image_stack (np.array): A 4D numpy array of shape(points, rows, columns, channels)
        crop_size (int): Size of the crop to take from the image. Assumes square crops

    Outputs:
        cropped_images (np.array): A 4D numpy array of shape (crops, rows, columns, channels)"""

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

    The tiled crops can overlap each other, with the overlap between tiles determined by
    the stride fraction. A stride fraction of 0.333 will move the window over 1/3 of
    the crop_size in x and y at each step, whereas a stride fraction of 1 will move
    the window the entire crop size at each iteration.

    Inputs:
        image_stack (np.array): A 4D numpy array of shape(points, rows, columns, channels)
        crop_size (int): size of the crop to take from the image. Assumes square crops
        stride_fraction (float): the relative size of the stride for overlapping
            crops as a function of the crop size.
    Outputs:
        cropped_images (np.array): A 4D numpy array of shape(crops, rows, cols, channels)"""

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

    Inputs
        dir_path: path to directory containing the sub directories

    Outputs
        None"""

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
