import os
import math
import warnings

import skimage.io as io
import numpy as np
import xarray as xr

import skimage.filters.rank as rank
import scipy.ndimage as nd


# data loading
def save_deepcell_tifs(model_output_xr, save_path, transform='pixel', points=None, watershed_smooth=3, pixel_smooth=[5]):
    """Extract and save tifs from deepcell output and save in directory format

        Args
            model_output: xarray of tifs output by deepcell
            save_path: folder to save tifs
            transform: one of pixel, fgbg, watershed, which determines how to process/save image
            points: optional list of points to extract. If none extracts all
            watershed_smooth: side length for square selem used for median smoothing
            pixel_smooth: variance used for gaussian filter smoothing
            """

    if len(model_output_xr.shape) != 4:
        raise ValueError("xarray data has the wrong dimensions, expecting 4")

    if points is None:
        points = model_output_xr.coords['points']
    else:
        if np.any(~np.isin(points, model_output_xr.coords['points'])):
            raise ValueError("Incorrect list of points given, not all are present in data structure")

    if type(pixel_smooth) is int:
        pixel_smooth = [pixel_smooth]
    elif type(pixel_smooth) is list:
        # this is good
        pass
    else:
        raise ValueError("pixel smooth is not a list or an integer")

    # keep only the selected points
    if len(points) == 1:
        # don't subset, will change dimensions
        pass
    else:
        model_output_xr = model_output_xr.loc[points, :, :, :]

    if transform == 'watershed':
        if model_output_xr.shape[-1] != 1:
            raise ValueError("Watershed transform selected, but last dimension is not 4")
        if model_output_xr.coords['masks'].values[0] != 'watershed_argmax':
            raise ValueError("Watershed transform selected, but first channel is not watershed_argmax")

        # create array to hold argmax and smoothed argmax mask
        watershed_processed = np.zeros(model_output_xr.shape[:-1] + (2, ), dtype='int8')
        watershed_processed[:, :, :, 0] = model_output_xr.values[:, :, :, 0]

        for i in range(model_output_xr.shape[0]):
            smoothed_argmax = rank.median(model_output_xr[i, :, :, 0], np.ones((watershed_smooth, watershed_smooth)))
            watershed_processed[i, :, :, 1] = smoothed_argmax

            # ignore low-contrast image warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(os.path.join(save_path, model_output_xr.coords['points'].values[i] +
                                       '_watershed.tiff'), watershed_processed[i, :, :, 0].astype('int8'))

                io.imsave(os.path.join(save_path, model_output_xr.coords['points'].values[i] +
                                       '_watershed_smoothed.tiff'), watershed_processed[i, :, :, 1].astype('int8'))

        mask = ["watershed", "watershed_smoothed"]
        watershed_processed_xr = xr.DataArray(watershed_processed, name=model_output_xr.name + "_processed",
                                           coords=[model_output_xr.coords['points'], range(model_output_xr.shape[1]),
                                                   range(model_output_xr.shape[2]), mask],
                                           dims=["points", "rows", "cols", "masks"])
        watershed_processed_xr.to_netcdf(os.path.join(save_path, watershed_processed_xr.name + '.nc'))

    elif transform == 'pixel':
        if model_output_xr.shape[-1] != 3:
            raise ValueError("pixel transform selected, but last dimension is not three")
        if model_output_xr.coords['masks'].values[0] != 'border':
            raise ValueError("pixel transform selected, but mask names don't match")
        pixel_proccesed_dim = 1 + len(pixel_smooth) + 1
        pixel_processed = np.zeros(model_output_xr.shape[:-1] + (pixel_proccesed_dim, ), dtype='int8')
        pixel_processed[:, :, :, 0:2] = model_output_xr.loc[:, :, :, ['border', 'interior']].values

        for i in range(model_output_xr.shape[0]):
            # smooth interior probability for each point
            for smooth in range(len(pixel_smooth)):
                # smooth output according to smooth value, save sequentially in xarray

                # ignore low-contrast image warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    smoothed_int = nd.gaussian_filter(model_output_xr[i, :, :, 1], pixel_smooth[smooth])

                pixel_processed[i, :, :, 2 + smooth] = smoothed_int

        # save output to xarray
        mask_labels = ["pixel_border", "pixel_interior"]
        for smooth in pixel_smooth:
            mask_labels.append("pixel_interior_smoothed_{}".format(smooth))

        pixel_processed_xr = xr.DataArray(pixel_processed, name=model_output_xr.name + "_processed",
                                            coords=[model_output_xr.coords['points'], range(model_output_xr.shape[1]),
                                                    range(model_output_xr.shape[2]), mask_labels],
                                            dims=["points", "rows", "cols", "masks"])

        # save processed masks for viewing
        for point in range(pixel_processed_xr.shape[0]):
            for mask in range(pixel_processed_xr.shape[-1]):
                io.imsave(os.path.join(save_path, "{}_{}.tiff".format(pixel_processed_xr.points[point].values,
                                                                 pixel_processed_xr.masks[mask].values)),
                            pixel_processed_xr[point, :, :, mask].values)

        pixel_processed_xr.to_netcdf(os.path.join(save_path, pixel_processed_xr.name + '.nc'))


def load_tifs_from_points_dir(point_dir, tif_folder=None, points=None, tifs=None, dtype="int16"):
    """Takes a set of TIFs from a directory structure organised by points, and loads them into a numpy array.

        Args:
            point_dir: directory path to points
            tif_folder: optional name of tif_folder within each point, otherwise assumes tifs are in Point folder
            points: optional list of point_dirs to load, otherwise loads all folders with Point in name
            tifs: optional list of TIFs to load, otherwise loads all TIFs
            dtype: dtype of array which will be used to store values

        Returns:
            Numpy array with shape [points, tifs, x_dim, y_dim]
    """

    if not os.path.isdir(point_dir):
        raise ValueError("Directory does not exist")

    if points is None:
        # get all point folders
        points = os.listdir(point_dir)
        points = [point for point in points if 'Point' in point]
        points = [point for point in points if os.path.isdir(os.path.join(point_dir, point))]
    else:
        # use supplied list, but check to make sure they all exist
        for point in points:
            if not os.path.isdir(os.path.join(point_dir, point)):
                raise ValueError("Could not find point folder {}".format(point))

    if len(points) == 0:
        raise ValueError("No points found in directory")

    # check to make sure tif subfolder name within point directory is correct
    if tif_folder is not None:
        if not os.path.isdir(os.path.join(point_dir, points[0], tif_folder)):
            raise ValueError("Invalid tif folder name")
    else:
        # no tif folder, change to empty string to read directly from base folder
        tif_folder = ""

    # get tifs from first point directory if no tif names supplied
    if tifs is None:
        tifs = os.listdir(os.path.join(point_dir, points[0], tif_folder))
        tifs = [tif for tif in tifs if '.tif' in tif]

    if len(tifs) == 0:
        raise ValueError("No tifs found in designated folder")

    # check to make sure supplied tifs exist
    for tif in tifs:
        if not os.path.isfile(os.path.join(point_dir, points[0], tif_folder, tif)):
            raise ValueError("Could not find {} in supplied directory {}".format(tif, os.path.join(point_dir, points[0], tif_folder, tif)))

    test_img = io.imread(os.path.join(point_dir, points[0], tif_folder, tifs[0]))
    img_data = np.zeros((len(points), test_img.shape[0], test_img.shape[1], len(tifs)), dtype=dtype)

    for point in range(len(points)):
        for tif in range(len(tifs)):
            img_data[point, :, :, tif] = io.imread(os.path.join(point_dir, points[point], tif_folder, tifs[tif]))

    # check to make sure that dtype wasn't too small for range of data
    if np.min(img_data) < 0:
        raise ValueError("Integer overflow from loading TIF image, try a larger dtype")

    # remove .tif or .tiff from image name
    tif_names = [os.path.splitext(tif)[0] for tif in tifs]

    img_xr = xr.DataArray(img_data, coords=[points, range(test_img.shape[0]), range(test_img.shape[1]), tif_names],
                          dims=["points", "rows", "cols", "channels"])

    return img_xr


def create_blank_channel(img_size, grid_size):
    """Creates a blank TIF of a given size that has a small number of positive pixels to avoid divide by zero errors
    Inputs:
        img_size: tuple specifying the size of the image to create
        grid_size: int that determines how many pieces to randomize within

    Outputs:
        blank_arr: a (mostly) blank array with positive pixels in random values
    """

    blank = np.zeros(img_size, dtype="int16")
    row_step = math.floor(blank.shape[0] / grid_size)
    col_step = math.floor(blank.shape[1] / grid_size)

    for row in range(grid_size):
        for col in range(grid_size):
            row_rand = np.random.randint(0, grid_size - 1)
            col_rand = np.random.randint(0, grid_size - 1)
            blank[row * row_step + row_rand, col * col_step + col_rand] = np.random.randint(15)

    return blank


def reorder_xarray_channels(channel_order, channel_xr, non_blank_channels=None):

    """Adds blank channels or changes the order of existing channels to match the ordering given by channel_order list
    Inputs:
        channel_order: list of channel names, which dictates final order of output xarray
        channel_xr: xarray containing the channel data for the available channels
        non_blank_channels: optional list of channels which aren't missing, and hence won't be replaced with blank tif:
            if not supplied, will default to assuming all channels in channel_order

    Outputs:
        xarray with the supplied channels in channel order"""

    if non_blank_channels is None:
        non_blank_channels = channel_order

    # error checking
    channels_in_xr = np.isin(non_blank_channels, channel_xr.channels)
    if len(channels_in_xr) != np.sum(channels_in_xr):
        bad_chan = non_blank_channels[np.where(~channels_in_xr)[0][0]]
        raise ValueError("{} was listed as a non-blank channel, but it is not in the channel xarray".format(bad_chan))

    channels_in_order = np.isin(non_blank_channels, channel_order)
    if len(channels_in_order) != np.sum(channels_in_order):
        bad_chan = non_blank_channels[np.where(~channels_in_order)[0][0]]
        raise ValueError("{} was listed as a non-blank channel, but it is not in the channel order".format(bad_chan))

    vals, counts = np.unique(channel_order, return_counts=True)
    duplicated = np.where(counts > 1)
    if len(duplicated[0] > 0):
        print("The following channels are duplicated: {}".format(vals[duplicated[0]]))

    # create array to hold all channels, including blank ones
    full_array = np.zeros((channel_xr.shape[:3] + (len(channel_order),)), dtype='int8')

    for i in range(len(channel_order)):
        if channel_order[i] in non_blank_channels:
            full_array[:, :, :, i] = channel_xr.loc[:, :, :, channel_order[i]].values
        else:
            im_crops = channel_xr.shape[1] // 32
            blank = create_blank_channel(channel_xr.shape[1:3], im_crops)
            full_array[:, :, :, i] = blank

    channel_xr_blanked = xr.DataArray(full_array, coords=[channel_xr.points, range(channel_xr.shape[1]),
                                                          range(channel_xr.shape[2]), channel_order],
                                   dims=["points", "rows", "cols", "channels"])

    return channel_xr_blanked


def combine_xarrays(xarrays, axis):
    """Combines a number of xarrays together

    Inputs:
        xarrays: a tuple of xarrays
        axis: either 0, if the xarrays will combined over different points, or -1, if they will be combined over channels

    Outputs:
        combined_xr: an xarray that is the combination of all inputs"""

    first_xr = xarrays[0]
    np_arr = first_xr.values

    # define iterator to hold coord values of dimension that is being stacked
    if axis == 0:
        iterator = first_xr.points.values
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
                raise ValueError("xarrays have different channel names")
        else:
            if not np.array_equal(cur_xr.points, first_xr.points):
                raise ValueError("xarrays have different point names")

        np_arr = np.concatenate((np_arr, cur_arr), axis=axis)
        if axis == 0:
            iterator = np.append(iterator, cur_xr.points.values)
        else:
            iterator = np.append(iterator, cur_xr.channels.values)

    # assign iterator to appropriate coord label
    if axis == 0:
        points = iterator
        channels = first_xr.channels.values
    else:
        points = first_xr.points
        channels = iterator

    combined_xr = xr.DataArray(np_arr, coords=[points, range(first_xr.shape[1]), range(first_xr.shape[2]), channels],
                               dims=["points", "rows", "cols", "channels"])

    return combined_xr


def crop_helper(image_stack, crop_size):
    """"Helper function to take an image, and return crops of size crop_size

    Inputs:
        image_stack (np.array): A 4D numpy array of shape(points, rows, columns, channels)
        crop_size (int): Size of the crop to take from the image. Assumes square crops

    Outputs:
        cropped_images (np.array): A 4D numpy array of shape (crops, rows, columns, channels)"""

    if len(image_stack.shape) != 4:
        raise ValueError("Incorrect dimensions of input image. Expecting 3D, got {}".format(image_stack.shape))

    # figure out number of crops for final image
    crop_num_row = math.ceil(image_stack.shape[1] / crop_size)
    crop_num_col = math.ceil(image_stack.shape[2] / crop_size)
    cropped_images = np.zeros((crop_num_row * crop_num_col * image_stack.shape[0], crop_size, crop_size,
                               image_stack.shape[3]), dtype="int16")

    # Determine if image will need to be padded with zeros due to uneven division by crop
    if image_stack.shape[1] % crop_size != 0 or image_stack.shape[2] % crop_size != 0:
        # create new array that is padded by one crop size on image dimensions
        new_shape = image_stack.shape[0], image_stack.shape[1] + crop_size, image_stack.shape[2] + crop_size, image_stack.shape[3]
        new_stack = np.zeros(new_shape, dtype="int16")
        new_stack[:, :image_stack.shape[1], :image_stack.shape[2], :] = image_stack
        image_stack = new_stack

    # iterate through the image row by row, cropping based on identified threshold
    img_idx = 0
    for point in range(image_stack.shape[0]):
        for row in range(crop_num_row):
            for col in range(crop_num_col):
                cropped_images[img_idx, :, :, :] = image_stack[point, (row * crop_size):((row + 1) * crop_size),
                                                       (col * crop_size):((col + 1) * crop_size), :]
                img_idx += 1

    return cropped_images


def crop_image_stack(image_stack, crop_size, stride_fraction):
    """Function to generate a series of tiled crops across an image. The tiled crops can overlap each other, with the
       overlap between tiles determined by the stride fraction. A stride fraction of 0.333 will move the window over
       1/3 of the crop_size in x and y at each step, whereas a stride fraction of 1 will move the window the entire crop
       size at each iteration.

    Inputs:
        image_stack (np.array): A 4D numpy array of shape(points, rows, columns, channels)
        crop_size (int): size of the crop to take from the image. Assumes square crops
        stride_fraction (float): the relative size of the stride for overlapping crops as a function of
        the crop size.
    Outputs:
        cropped_images (np.array): A 4D numpy array of shape(crops, rows, cols, channels)"""

    if len(image_stack.shape) != 4:
        raise ValueError("Incorrect dimensions of input image. Expecting 3D, got {}".format(image_stack.shape))

    if crop_size > image_stack.shape[1]:
        raise ValueError("Invalid crop size: img shape is {} and crop size is {}".format(image_stack.shape, crop_size))

    if stride_fraction > 1:
        raise ValueError("Invalid stride fraction. Must be less than 1, passed a value of {}".format(stride_fraction))

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
                img_shift = image_stack[:, (row_shift * stride_step):, (col_shift * stride_step):, :]
                # print("shape of the input image is {}".format(img_shift.shape))
                temp_images = crop_helper(img_shift, crop_size)
                cropped_images = np.append(cropped_images, temp_images, axis=0)

    return cropped_images

