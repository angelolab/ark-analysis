import os

import numpy as np
import skimage.io as io

from skimage import morphology
from skimage.measure import label
from scipy.ndimage import gaussian_filter

from alpineer import io_utils, misc_utils, load_utils
from ark.utils import data_utils


def create_mask(arr, intensity_thresh, sigma=2, min_mask_size=0, max_hole_size=1000):
    """Generates a binary mask from a signal image

    Args:
        arr (np.ndarray): array to be masked
        intensity_thresh (float): threshold for the array values to use for masking
        sigma (float): sigma for gaussian blur
        min_mask_size (int): minimum size of masked objects to include
        max_hole_size (int): maximum size of holes to leave in masked objects
    """
    # create a binary mask
    img_smoothed = gaussian_filter(arr.astype(float), sigma=sigma)
    img_mask = img_smoothed > intensity_thresh

    # if no post-processing return as is
    if min_mask_size == 0:
        return img_mask

    # otherwise, clean up the mask before returning
    label_mask = label(img_mask)
    label_mask = morphology.remove_small_objects(label_mask, min_size=min_mask_size)
    label_mask = morphology.remove_small_holes(label_mask, area_threshold=max_hole_size)

    return label_mask


def generate_img_masks(img_dir, mask_dir, channels, mask_name):
    """Creates a single signal mask for each FOV when given the channels to aggregate.

    Args:
        img_dir (str): path to the image tiff directory
        mask_dir (str): path where the masks will be saved
        channels (list): list of channels to combine to create a single mask for
        mask_name (str): name for the new mask file created
    """

    fovs = io_utils.list_folders(img_dir)

    # check valid channel name
    channel_list = io_utils.remove_file_extensions(io_utils.list_files(fovs[0]))
    misc_utils.verify_in_list(input_channels=channels, all_channels=channel_list)

    for fov in fovs:
        test_img = io.imread(os.path.join(img_dir, fov, channel_list[0] + 'tiff'))

        # if more than one channel, create composite image
        img = np.zeros_like(test_img)
        for chan in channels:
            single_chan_img = io.imread(os.path.join(img_dir, chan + '.tiff'))
            img = img + single_chan_img

        # create mask
        mask = create_mask(img, intensity_thresh=350, sigma=2, min_mask_size=5000, max_hole_size=1000)

        # save mask
        save_dir = os.path.join(mask_dir, fov)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_utils.save_fov_mask(mask_name, save_dir, mask)


def create_cell_mask(seg_mask, cell_table, fov_name, cell_types, sigma=10, smooth_thresh=0.3,
                     min_mask_size=0, max_hole_size=100000):
    """Generates a binary from the cells listed in `cell_types`

    args:
        seg_mask (numpy.ndarray): segmentation mask
        cell_table (pandas.DataFrame): cell table containing segmentation IDs and cell types
        fov_name (str): name of the fov to process
        cell_types (list): list of cell types to include in the mask
        sigma (float): sigma for gaussian smoothing
        smooth_thresh (float): threshold for including a pixel in the smoothed mask
        min_mask_size (int): minimum size of a mask to include
        max_hole_size (int): maximum size of a hole to leave without filling

    returns:
        numpy.ndarray: binary mask
    """
    # get cell labels for fov and cell type
    cell_subset = cell_table[cell_table['fov'] == fov_name]
    cell_subset = cell_subset[cell_subset['cell_cluster_broad'].isin(cell_types)]
    cell_labels = cell_subset['label'].values

    # create mask for cell type
    cell_mask = np.isin(seg_mask, cell_labels)

    # postprocess mask
    cell_mask = create_mask(arr=cell_mask, intensity_thresh=smooth_thresh,
                            sigma=sigma, min_mask_size=min_mask_size, max_hole_size=max_hole_size)

    return cell_mask


def generate_cell_masks(seg_dir, mask_dir, cell_table, cell_types, mask_name):
    """Creates a single cell mask for each FOV when given the cell types to aggregate.

    Args:
        seg_dir (str): path to the cell segmentation tiff directory
        mask_dir (str): path where the masks will be saved
        cell_table (pd.DataFrame): Dataframe containing all cell labels and their cell type
        cell_types (list): list of cell phenotypes that will be used to create the mask
        mask_name (str): name for the new mask file created
    """

    fovs = io_utils.remove_file_extensions(io_utils.list_files(seg_dir, 'whole_cell'))

    for fov in fovs:
        fov_name = fov.replace()

        seg_mask = io.imread(os.path.join(mask_dir, fov))

        # create mask
        mask = create_cell_mask(seg_mask, cell_table, fov_name, cell_types)

        # save mask
        save_dir = os.path.join(mask_dir, fov)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        data_utils.save_fov_mask(mask_name, save_dir, mask)
