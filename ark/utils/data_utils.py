import math
import os
import pathlib
import shutil
from typing import List, Union

import datasets
import feather
import numpy as np
import skimage.io as io
import xarray as xr
from tqdm.notebook import tqdm_notebook as tqdm

from ark import settings
from ark.utils import load_utils
from ark.utils.misc_utils import verify_in_list


def save_fov_images(fovs, data_dir, img_xr, sub_dir=None, name_suffix=''):
    """Given an xarray of a fov image, saves it

    Args:
        fovs (list):
            The list of fovs to save in `img_xr`
        data_dir (str):
            The directory to save the images
        img_xr (xarray.DataArray):
            The array of images per fov
        sub_dir (Optional[str]):
            The subdirectory to save the images in. If specified images are saved to
            "data_dir/sub_dir". If `sub_dir = None` the images are saved to "data_dir". Defaults
            to None.
        name_suffix (str):
            Specify what to append at the end of every fov.
    """

    img_xr = img_xr.astype('int16')

    if not os.path.exists(data_dir):
        raise FileNotFoundError("data_dir %s does not exist" % data_dir)

    # verify that the fovs provided are valid
    verify_in_list(
        provided_fovs=fovs,
        img_xr_fovs=img_xr.fovs.values
    )

    # ensure None is handled correctly in file path generation
    if sub_dir is None:
        sub_dir = ''

    save_dir = os.path.join(data_dir, sub_dir)

    # make the save_dir if it doesn't already exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # retrieve the image for each fov
    for fov in fovs:
        fov_img_data = img_xr.loc[fov, ...].values

        # define the file name as the fov name with the name suffix appended
        fov_file = fov + name_suffix + '.tiff'

        # save the image to data_dir
        io.imsave(os.path.join(save_dir, fov_file), fov_img_data, check_contrast=False)


def label_cells_by_cluster(fovs, all_data, label_maps, fov_col=settings.FOV_ID,
                           cell_label_column=settings.CELL_LABEL,
                           cluster_column=settings.KMEANS_CLUSTER):
    """Translates cell-ID labeled images according to the clustering assignment.

    Takes a list of fovs, and relabels each image (array) according to the assignment
    of cell IDs to cluster label.

    Args:
        fovs (list):
            List of fovs to relabel.
        all_data (pandas.DataFrame):
            data including fovs, cell labels, and cell expression matrix for all markers.
        label_maps (xarray.DataArray):
            xarray of label maps for multiple fovs
        fov_col (str):
            column with the fovs names in all_data.
        cell_label_column (str):
            column with the cell labels in all_data.
        cluster_column (str):
            column with the cluster labels in all_data.

    Returns:
        xarray.DataArray:
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

    return xr.DataArray(img_data, coords=[fovs, range(img_data[0].shape[0]),
                                          range(img_data[0].shape[1])],
                        dims=["fovs", "rows", "cols"])


def generate_cell_cluster_mask(fov, base_dir, seg_dir, cell_data_name,
                               cell_cluster_col='cell_meta_cluster', seg_suffix='_feature_0.tif'):
    """For a fov, create a mask labeling each cell with their SOM or meta cluster label

    Args:
        fov (list):
            The fov to relabel
        base_dir (str):
            The path to the data directory
        seg_dir (str):
            The path to the segmentation data
        cell_data_name (str):
            The path to the cell data with both cell SOM and meta cluster assignments
        cell_cluster_col (str):
            Whether to assign SOM or meta clusters.
            Needs to be `'cell_som_cluster'` or `'cell_meta_cluster'`
        seg_suffix (str):
            The suffix that the segmentation images use

    Returns:
        xarray.DataArray:
            The labeled images (dims: ["fovs", "rows", "cols"])
    """

    # path checking
    if not os.path.exists(seg_dir):
        raise FileNotFoundError("seg_dir %s does not exist" % seg_dir)

    if not os.path.exists(os.path.join(base_dir, cell_data_name)):
        raise FileNotFoundError(
            "Cell data file %s does not exist in base_dir %s" % (cell_data_name, base_dir))

    # verify the cluster_col provided is valid
    verify_in_list(
        provided_cluster_col=cell_cluster_col,
        valid_cluster_cols=['cell_som_cluster', 'cell_meta_cluster']
    )

    # load the consensus data in
    cell_consensus_data = feather.read_dataframe(os.path.join(base_dir, cell_data_name))

    # ensure the cluster col will be displayed as an integer and not a float
    cell_consensus_data[cell_cluster_col] = cell_consensus_data[cell_cluster_col].astype(int)

    # verify the fov is valid
    verify_in_list(
        provided_fovs=fov,
        consensus_fovs=cell_consensus_data['fov']
    )

    # define the file for whole cell
    whole_cell_files = [fov + seg_suffix]

    # load the segmentation labels in
    label_maps = load_utils.load_imgs_from_dir(data_dir=seg_dir,
                                               files=whole_cell_files,
                                               xr_dim_name='compartments',
                                               xr_channel_names=['whole_cell'],
                                               trim_suffix=seg_suffix.split('.')[0])

    # use label_cells_by_cluster to create cell masks
    img_data = label_cells_by_cluster(
        [fov], cell_consensus_data, label_maps, fov_col='fov',
        cell_label_column='segmentation_label', cluster_column=cell_cluster_col
    )

    return img_data


def generate_pixel_cluster_mask(fov, base_dir, tiff_dir, chan_file_path,
                                pixel_data_dir, pixel_cluster_col='pixel_meta_cluster'):
    """For a fov, create a mask labeling each pixel with their SOM or meta cluster label

    Args:
        fov (list):
            The fov to relabel
        base_dir (str):
            The path to the data directory
        tiff_dir (str):
            The path to the tiff data
        chan_file_path (str):
            The path to the sample channel file to load (`tiff_dir` as root).
            Used to determine dimensions of the pixel mask.
        pixel_data_dir (str):
            The path to the data with full pixel data.
            This data should also have the SOM and meta cluster labels appended.
        pixel_cluster_col (str):
            Whether to assign SOM or meta clusters
            needs to be `'pixel_som_cluster'` or `'pixel_meta_cluster'`

    Returns:
        xarray.DataArray:
            The labeled images (dims: ["fovs", "rows", "cols"])
    """

    # path checking
    if not os.path.exists(tiff_dir):
        raise FileNotFoundError("tiff_dir %s does not exist")

    if not os.path.exists(os.path.join(tiff_dir, chan_file_path)):
        raise FileNotFoundError("chan_file_path %s does not exist in tiff_dir %s"
                                % (chan_file_path, tiff_dir))

    if not os.path.exists(os.path.join(base_dir, pixel_data_dir)):
        raise FileNotFoundError(
            "Pixel data dir %s does not exist in base_dir %s" % (pixel_data_dir, base_dir)
        )

    # verify the pixel_cluster_col provided is valid
    verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster']
    )

    # verify the fov is valid
    verify_in_list(
        provided_fov_file=[fov + '.feather'],
        consensus_fov_files=os.listdir(os.path.join(base_dir, pixel_data_dir))
    )

    # read the sample channel file to determine size of pixel cluster mask
    channel_data = np.squeeze(io.imread(os.path.join(tiff_dir, chan_file_path)))

    # define an array to hold the overlays for the fov
    img_data = np.zeros((1, channel_data.shape[0], channel_data.shape[1]))

    fov_data = feather.read_dataframe(
        os.path.join(base_dir, pixel_data_dir, fov + '.feather')
    )

    # ensure integer display and not float
    fov_data[pixel_cluster_col] = fov_data[pixel_cluster_col].astype(int)

    # get the pixel coordinates
    x_coords = fov_data['row_index'].values
    y_coords = fov_data['column_index'].values

    # convert to 1D indexing
    coordinates = x_coords * img_data.shape[1] + y_coords

    # get the cooresponding cluster labels for each pixel
    cluster_labels = list(fov_data[pixel_cluster_col])

    # assign each coordinate in pixel_cluster_mask to its respective cluster label
    img_subset = img_data[0, ...].ravel()
    img_subset[coordinates] = cluster_labels
    img_data[0, ...] = img_subset.reshape(img_data[0, ...].shape)

    # create the stacked img_data xarray and return
    return xr.DataArray(img_data, coords=[[fov], range(img_data[0].shape[0]),
                                          range(img_data[0].shape[1])],
                        dims=["fovs", "rows", "cols"])


def generate_and_save_pixel_cluster_masks(fovs: List[str],
                                          base_dir: Union[pathlib.Path, str],
                                          save_dir: Union[pathlib.Path, str],
                                          tiff_dir: Union[pathlib.Path, str],
                                          chan_file: Union[pathlib.Path, str],
                                          pixel_data_dir: Union[pathlib.Path, str],
                                          pixel_cluster_col: str = 'pixel_meta_cluster',
                                          sub_dir: str = None,
                                          name_suffix: str = ''):
    """Generates pixel cluster masks and saves them in batches for downstream analysis.

    Args:
        fovs (List[str]):
            A list of fovs to generate and save pixel masks for.
        base_dir (Union[pathlib.Path, str]):
            The path to the data directory.
        save_dir (Union[pathlib.Path, str]):
            The directory to save the generated pixel cluster masks.
        tiff_dir (Union[pathlib.Path, str]):
            The path to the directory with the tiff data.
        chan_file (Union[pathlib.Path, str]):
            The path to the channel file inside each FOV folder (FOV folder as root).
            Used to determine dimensions of the pixel mask.
        pixel_data_dir (Union[pathlib.Path, str]):
            The path to the data with full pixel data.
            This data should also have the SOM and meta cluster labels appended.
        pixel_cluster_col (str, optional):
            The path to the data with full pixel data.
            This data should also have the SOM and meta cluster labels appended.
            Defaults to 'pixel_meta_cluster'.
        sub_dir (str, optional):
            The subdirectory to save the images in. If specified images are saved to
            "data_dir/sub_dir". If `sub_dir = None` the images are saved to "data_dir". Defaults
            to None.
        name_suffix (str, optional):
            Specify what to append at the end of every pixel mask. Defaults to ''.
    """

    # create the pixel cluster masks over each fov batch.
    with tqdm(total=len(fovs), desc="Pixel Cluster Mask Generation") as pixel_mask_progress:
        for fov in fovs:
            # define the path to provided channel file in base_dir, used to calculate dimensions
            chan_file_path = os.path.join(fov, chan_file)

            pixel_masks: xr.DataArray =\
                generate_pixel_cluster_mask(fov=fov, base_dir=base_dir, tiff_dir=tiff_dir,
                                            chan_file_path=chan_file_path,
                                            pixel_data_dir=pixel_data_dir,
                                            pixel_cluster_col=pixel_cluster_col)

            save_fov_images([fov], data_dir=save_dir, img_xr=pixel_masks, sub_dir=sub_dir,
                            name_suffix=name_suffix)

            pixel_mask_progress.update(1)


def generate_and_save_cell_cluster_masks(fovs: List[str],
                                         base_dir: Union[pathlib.Path, str],
                                         save_dir: Union[pathlib.Path, str],
                                         seg_dir: Union[pathlib.Path, str],
                                         cell_data_name: Union[pathlib.Path, str],
                                         cell_cluster_col: str = 'cell_meta_cluster',
                                         seg_suffix: str = '_feature_0.tif',
                                         sub_dir: str = None,
                                         name_suffix: str = ''):
    """Generates cell cluster masks and saves them in batches for downstream analysis.

    Args:
        fovs (List[str]):
            A list of fovs to generate and save pixel masks for.
        base_dir (Union[pathlib.Path, str]):
            The path to the data directory.
        save_dir (Union[pathlib.Path, str]):
            The directory to save the generated cell cluster masks.
        seg_dir (Union[pathlib.Path, str]):
            The path to the segmentation data.
        cell_data_name (Union[pathlib.Path, str]):
            The path to the cell data with both cell SOM and meta cluster assignments
        cell_cluster_col (str, optional):
            Whether to assign SOM or meta clusters. Needs to be `'cell_som_cluster'` or
            `'cell_meta_cluster'`. Defaults to `'cell_meta_cluster'`.
        seg_suffix (str, optional):
            The suffix that the segmentation images use. Defaults to `'_feature_0.tif'`.
        sub_dir (str, optional):
            The subdirectory to save the images in. If specified images are saved to
            "data_dir/sub_dir". If `sub_dir = None` the images are saved to "data_dir".
            Defaults to None.
        name_suffix (str, optional):
            Specify what to append at the end of every cell mask. Defaults to ''.
    """

    # create the pixel cluster masks over each fov batch.
    with tqdm(total=len(fovs), desc="Cell Cluster Mask Generation") as cell_mask_progress:
        for fov in fovs:
            cell_masks: xr.DataArray =\
                generate_cell_cluster_mask(fov=fov, base_dir=base_dir, seg_dir=seg_dir,
                                           cell_data_name=cell_data_name,
                                           cell_cluster_col=cell_cluster_col,
                                           seg_suffix=seg_suffix)

            save_fov_images([fov], data_dir=save_dir, img_xr=cell_masks, sub_dir=sub_dir,
                            name_suffix=name_suffix)

            cell_mask_progress.update(1)


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
def generate_deepcell_input(data_dir, tiff_dir, nuc_channels, mem_channels, fovs,
                            is_mibitiff=False, img_sub_folder="TIFs", batch_size=5,
                            dtype="int16"):
    """Saves nuclear and membrane channels into deepcell input format.
    Either nuc_channels or mem_channels should be specified.

    Writes summed channel images out as multitiffs (channels first)

    Args:
        data_dir (str):
            location to save deepcell input tifs
        tiff_dir (str):
            directory containing folders of images, is_mibitiff determines what type
        nuc_channels (list):
            nuclear channels to be summed over
        mem_channels (list):
            membrane channels to be summed over
        fovs (list):
            list of folders to or MIBItiff files to load imgs from
        is_mibitiff (bool):
            if the images are of type MIBITiff
        img_sub_folder (str):
            if is_mibitiff is False, define the image subfolder for each fov
            ignored if is_mibitiff is True
        batch_size (int):
            the number of fovs to process at once for each batch
        dtype (str/type):
            optional specifier of image type.  Overwritten with warning for float images
    Raises:
        ValueError:
            Raised if nuc_channels and mem_channels are both None or empty
    """

    # cannot have no nuclear and no membrane channels
    if not nuc_channels and not mem_channels:
        raise ValueError('Either nuc_channels or mem_channels should be non-empty.')

    # define the channels list by combining nuc_channels and mem_channels
    channels = (nuc_channels if nuc_channels else []) + (mem_channels if mem_channels else [])

    # filter channels for None (just in case)
    channels = [channel for channel in channels if channel is not None]

    # define a list of fov batches to process over
    fov_batches = [fovs[i:i + batch_size] for i in range(0, len(fovs), batch_size)]

    for fovs in fov_batches:
        # load the images in the current fov batch
        if is_mibitiff:
            data_xr = load_utils.load_imgs_from_mibitiff(
                tiff_dir, mibitiff_files=fovs, channels=channels)
        else:
            data_xr = load_utils.load_imgs_from_tree(
                tiff_dir, img_sub_folder=img_sub_folder, fovs=fovs, channels=channels)

        # write each fov data to data_dir
        for fov in data_xr.fovs.values:
            out = np.zeros((2, data_xr.shape[1], data_xr.shape[2]), dtype=data_xr.dtype)

            # sum over channels and add to output
            if nuc_channels:
                out[0] = np.sum(data_xr.loc[fov, :, :, nuc_channels].values, axis=2)
            if mem_channels:
                out[1] = np.sum(data_xr.loc[fov, :, :, mem_channels].values, axis=2)

            save_path = os.path.join(data_dir, f"{fov}.tif")
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

            save_path = os.path.join(img_dir, names[i])
            io.imsave(save_path, channel, plugin='tifffile', check_contrast=False)


def download_example_data(save_dir: Union[str, pathlib.Path]):
    """Downloads the example dataset from Hugging Face Hub.
    The following is a link to the dataset used:
    https://huggingface.co/datasets/angelolab/ark_example

    The dataset will be saved in `{save_dir}/example_dataset/image_data`.

    Args:
        save_dir (Union[str, pathlib.Path]): The directory to save the example dataset in.
    """

    # Downloads the dataset
    ds = datasets.load_dataset("angelolab/ark_example")

    data_path = pathlib.Path(ds["base_dataset"]["Data Path"][0]) / "image_data"

    shutil.copytree(data_path, pathlib.Path(save_dir) / "image_data",
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('._*'))
