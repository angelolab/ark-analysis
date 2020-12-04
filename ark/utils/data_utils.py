import os
import math
import skimage.io as io
import numpy as np
import xarray as xr

from ark import settings
from ark.utils.misc_utils import verify_in_list


def label_cells_by_cluster(fovs, all_data, label_maps, fov_col=settings.FOV_ID,
                           cell_label_column=settings.CELL_LABEL,
                           cluster_column=settings.KMEANS_CLUSTER):
    """ Translates cell-ID labeled images according to the clustering assignment.

    Takes a list of fovs, and relabels each image (array) according to the assignment
    of cell IDs to cluster label.

    Args:
        fovs (list):
            List of fovs to relabel.
        all_data (pandas.DataFrame):
            data including fovs, cell labels, and cell expression matrix for all markers.
        label_maps (xr.DataArray):
            xarray of label maps for multiple fovs
        fov_col (str):
            column with the fovs names in all_data.
        cell_label_column (str):
            column with the cell labels in all_data.
        cluster_column (str):
            column with the cluster labels in all_data.
    Returns:
        xr.DataArray:
            The relabeled images (dims: ["fovs", "rows", "cols"]).
    """

    # check if included fovs found in fov_col
    verify_in_list(fov_names=fovs, all_data_fovs=all_data[fov_col].unique())
    verify_in_list(fov_names=fovs, label_map_fovs=label_maps.fovs.values)

    img_data = []
    for fov in fovs:
        df = all_data[all_data[fov_col] == fov]
        labels_dict = dict(zip(df[cell_label_column], df[cluster_column]))
        labeled_img_array = label_maps.loc[label_maps.fovs == fov].squeeze().values
        relabeled_img_array = relabel_segmentation(labeled_img_array, labels_dict)
        img_data.append(relabeled_img_array)
    np.stack(img_data, axis=0)
    return xr.DataArray(img_data, coords=[fovs, range(img_data[0].shape[0]),
                                          range(img_data[0].shape[1])],
                        dims=["fovs", "rows", "cols"])


def relabel_segmentation(labeled_image, labels_dict):
    """Takes a labeled image and translates its labels according to a dictionary.

    Returns the relabeled array (according to the dictionary).

    Args:
        labeled_image (numpy.ndarray):
            2D numpy array of labeled cell objects.
        labels_dict (dict):
            a mapping between labeled cells and their clusters.
    Returns:
        numpy.ndarray:
            The relabeled array.
    """

    img = np.copy(labeled_image)
    unique_cell_ids = np.unique(labeled_image)
    unique_cell_ids = unique_cell_ids[np.nonzero(unique_cell_ids)]

    default_label = max(labels_dict.values()) + 1
    for cell_id in unique_cell_ids:
        img[labeled_image == cell_id] = labels_dict.get(cell_id, default_label)
    return img


# TODO: Add metadata for channel name (eliminates need for fixed-order channels)
def generate_deepcell_input(data_xr, data_dir, nuc_channels, mem_channels):
    """Saves nuclear and membrane channels into deepcell input format.
    Either nuc_channels or mem_channels should be specified.

    Writes summed channel images out as multitiffs (channels first)

    Args:
        data_xr (xr.DataArray):
            xarray containing nuclear and membrane channels over many fov's
        data_dir (str):
            location to save deepcell input tifs
        nuc_channels (list):
            nuclear channels to be summed over
        mem_channels (list):
            membrane channels to be summed over
    Raises:
        ValueError:
            Raised if nuc_channels and mem_channels are both None or empty
    """

    if not nuc_channels and not mem_channels:
        raise ValueError('Either nuc_channels or mem_channels should be non-empty.')

    for fov in data_xr.fovs.values:
        out = np.zeros((2, data_xr.shape[1], data_xr.shape[2]), dtype=data_xr.dtype)

        # sum over channels and add to output
        if nuc_channels:
            out[0] = np.sum(data_xr.loc[fov, :, :, nuc_channels].values, axis=2)
        if mem_channels:
            out[1] = np.sum(data_xr.loc[fov, :, :, mem_channels].values, axis=2)

        save_path = os.path.join(data_dir, f'{fov}.tif')
        io.imsave(save_path, out, plugin='tifffile', check_contrast=False)


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
                               dims=['fovs', 'rows', 'cols', 'channels'])
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
