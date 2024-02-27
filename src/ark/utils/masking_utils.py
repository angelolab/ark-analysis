import os

import numpy as np
import skimage.io as io

from skimage import morphology
from skimage.measure import label
from scipy.ndimage import gaussian_filter

from alpineer import io_utils, misc_utils, load_utils
from ark.utils import data_utils


def create_mask(arr, intensity_thresh, sigma, min_mask_size, max_hole_size):
    """Generates a binary mask from a signal image

    Args:
        arr (np.ndarray): array to be masked
        intensity_thresh (float): threshold for the array values to use for masking
        sigma (float): sigma for gaussian blur
        min_mask_size (int): minimum size of masked objects to include
        max_hole_size (int): maximum size of holes to leave in masked objects
    """
    # create a binary mask
    arr_smoothed = gaussian_filter(arr.astype(float), sigma=sigma)
    arr_mask = (arr_smoothed > intensity_thresh).astype(int)

    # if no post-processing return as is
    if min_mask_size == 0:
        return arr_mask

    # otherwise, clean up the mask before returning
    label_mask = label(arr_mask)
    label_mask = morphology.remove_small_objects(label_mask, min_size=min_mask_size)
    label_mask = morphology.remove_small_holes(label_mask, area_threshold=max_hole_size)

    return label_mask


def create_composite_image(img_dir, fov, channels):
    """Read in specified channel tiffs and create a composite image.
    Args:
        img_dir (str): path to the image tiff directory
        fov (str): which FOV to use
        channels(list): list of channels to combine to create a single mask for

    Returns:
        numpy.ndarray: image array of aggregated signal
    """
    # check correct image directory path
    io_utils.validate_paths([img_dir])

    # create empty array of correct size
    imgs_arr = load_utils.load_imgs_from_tree(img_dir, fovs=[fov], channels=channels)

    # aggregate array along channel dimension
    composite_arr = imgs_arr.sum(dim="channels")

    return composite_arr


def generate_signal_masks(img_dir, mask_dir, channels, mask_name, intensity_thresh=350, sigma=2,
                          min_mask_size=5000, max_hole_size=1000):
    """Creates a single signal mask for each FOV when given the channels to aggregate.

    Args:
        img_dir (str): path to the image tiff directory
        mask_dir (str): path where the masks will be saved
        channels (list): list of channels to combine to create a single mask for
        mask_name (str): name for the new mask file created
        intensity_thresh (float): threshold for the array values to use for masking
        sigma (float): sigma for gaussian blur
        min_mask_size (int): minimum size of masked objects to include
        max_hole_size (int): maximum size of holes to leave in masked objects
    """
    # check correct image directory path
    io_utils.validate_paths([img_dir])
    fovs = io_utils.list_folders(img_dir)

    # check valid channel name
    channel_list = io_utils.remove_file_extensions(
        io_utils.list_files(os.path.join(img_dir, fovs[0])))
    misc_utils.verify_in_list(input_channels=channels, all_channels=channel_list)

    for fov in fovs:
        # create composite image (or read in single image)
        img = create_composite_image(img_dir, fov, channels)

        # create mask
        mask = create_mask(img, intensity_thresh, sigma, min_mask_size, max_hole_size)

        # save mask
        save_dir = os.path.join(mask_dir, fov)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_utils.save_fov_mask(mask_name, save_dir, mask)


def create_cell_mask(seg_mask, cell_table, fov_name, cell_types, cluster_col, sigma, smooth_thresh,
                     min_mask_size=0, max_hole_size=0):
    """Generates a binary from the cells listed in `cell_types`

    Args:
        seg_mask (numpy.ndarray): segmentation mask
        cell_table (pandas.DataFrame): cell table containing segmentation IDs and cell types
        fov_name (str): name of the fov to process
        cell_types (list): list of cell types to include in the mask
        cluster_col (str): column in cell table containing cell cluster
        sigma (float): sigma for gaussian smoothing
        smooth_thresh (float): threshold for including a pixel in the smoothed mask
        min_mask_size (int): minimum size of a mask to include, default 0
        max_hole_size (int): maximum size of a hole to leave without filling, default 0

    Returns:
        numpy.ndarray: binary mask
    """
    # get cell labels for fov and cell type
    cell_subset = cell_table[cell_table['fov'] == fov_name]
    cell_subset = cell_subset[cell_subset[cluster_col].isin(cell_types)]
    cell_labels = cell_subset['label'].values

    # create mask for cell type
    cell_mask = np.isin(seg_mask, cell_labels)

    # binarize and blur mask, no minimum size requirement or hole removal for cell masks
    cell_mask = create_mask(arr=cell_mask, intensity_thresh=smooth_thresh,
                            sigma=sigma, min_mask_size=min_mask_size, max_hole_size=max_hole_size)

    return cell_mask


def generate_cell_masks(seg_dir, mask_dir, cell_table, cell_types, cluster_col, mask_name, sigma=10,
                        smooth_thresh=0.3):
    """Creates a single cell mask for each FOV when given the cell types to aggregate.

    Args:
        seg_dir (str): path to the cell segmentation tiff directory
        mask_dir (str): path where the masks will be saved
        cell_table (pd.DataFrame): Dataframe containing all cell labels and their cell type
        cell_types (list): list of cell phenotypes that will be used to create the mask
        cluster_col (str): column in cell table containing cell cluster
        mask_name (str): name for the new mask file created
        sigma (float): sigma for gaussian smoothing
        smooth_thresh (float): threshold for including a pixel in the smoothed mask
    """

    fov_files = io_utils.list_files(seg_dir)

    for files in fov_files:
        fov_name = files.replace('_whole_cell.tiff', '')

        seg_mask = load_utils.load_imgs_from_dir(
            data_dir=seg_dir, files=[files], xr_dim_name='compartments',
            xr_channel_names=['whole_cell']
        )

        # create mask
        mask = create_cell_mask(
            np.array(seg_mask[0, :, :, 0]), cell_table, fov_name, cell_types, cluster_col, sigma,
            smooth_thresh)

        # save mask
        save_dir = os.path.join(mask_dir, fov_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        data_utils.save_fov_mask(mask_name, save_dir, mask)
