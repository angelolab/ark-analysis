import os
import pathlib
import re
from typing import List, Union

import feather
import natsort as ns
import numpy as np
import pandas as pd
import skimage.io as io
from alpineer import data_utils, image_utils, io_utils, load_utils, misc_utils
from tqdm.notebook import tqdm_notebook as tqdm

from ark import settings


def save_fov_mask(fov, data_dir, mask_data, sub_dir=None, name_suffix=''):
    """Saves a provided cluster label mask overlay for a FOV.

    Args:
        fov (str):
            The FOV to save
        data_dir (str):
            The directory to save the cluster mask
        mask_data (numpy.ndarray):
            The cluster mask data for the FOV
        sub_dir (Optional[str]):
            The subdirectory to save the masks in. If specified images are saved to
            "data_dir/sub_dir". If `sub_dir = None` the images are saved to `"data_dir"`.
            Defaults to `None`.
        name_suffix (str):
            Specify what to append at the end of every fov.
    """

    # data_dir validation
    io_utils.validate_paths(data_dir)

    # ensure None is handled correctly in file path generation
    if sub_dir is None:
        sub_dir = ''

    save_dir = os.path.join(data_dir, sub_dir)

    # make the save_dir if it doesn't already exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # define the file name as the fov name with the name suffix appended
    fov_file = fov + name_suffix + '.tiff'

    # save the image to data_dir
    image_utils.save_image(os.path.join(save_dir, fov_file), mask_data)


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

    # cast to int16 to allow for Photoshop loading
    relabeled_img = np.vectorize(
        lambda x: labels_dict.get(x, default_label) if x != 0 else 0
    )(img).astype('int16')

    return relabeled_img


def label_cells_by_cluster(fov, all_data, label_map, fov_col=settings.FOV_ID,
                           cell_label_column=settings.CELL_LABEL,
                           cluster_column=settings.KMEANS_CLUSTER):
    """Translates cell-ID labeled images according to the clustering assignment.

    Takes a single FOV, and relabels the image according to the assignment
    of cell IDs to cluster label.

    Args:
        fov (str):
            The FOV to relabel
        all_data (pandas.DataFrame):
            data including fovs, cell labels, and cell expression matrix for all markers.
        label_map (xarray.DataArray):
            label map for a single FOV
        fov_col (str):
            column with the fovs names in `all_data`.
        cell_label_column (str):
            column with the cell labels in `all_data`.
        cluster_column (str):
            column with the cluster labels in `all_data`.

    Returns:
        numpy.ndarray:
            The image with new designated label assignments
    """

    # verify that fov found in all_data
    # NOTE: label_map fov validation happens in loading function
    misc_utils.verify_in_list(fov_name=[fov], all_data_fovs=all_data[fov_col].unique())

    # subset all_data on the FOV
    df = all_data[all_data[fov_col] == fov]

    # generate the labels to use
    labels_dict = dict(zip(df[cell_label_column], df[cluster_column]))

    # condense extraneous axes
    labeled_img_array = label_map.squeeze().values

    # relabel the array
    relabeled_img_array = relabel_segmentation(labeled_img_array, labels_dict)

    return relabeled_img_array


def generate_cell_cluster_mask(fov, base_dir, seg_dir, cell_data,
                               cell_cluster_col='cell_meta_cluster',
                               seg_suffix='_whole_cell.tiff'):
    """For a fov, create a mask labeling each cell with their SOM or meta cluster label

    Args:
        fov (list):
            The fov to relabel
        base_dir (str):
            The path to the data directory
        seg_dir (str):
            The path to the segmentation data
        cell_data (pandas.DataFrame):
            The cell data with both cell SOM and meta cluster assignments
        cell_cluster_col (str):
            Whether to assign SOM or meta clusters.
            Needs to be `'cell_som_cluster'` or `'cell_meta_cluster'`
        seg_suffix (str):
            The suffix that the segmentation images use. Defaults to `'_whole_cell.tiff'`.

    Returns:
        numpy.ndarray:
            The image overlaid with cell cluster labels
    """

    # path checking
    io_utils.validate_paths([seg_dir])

    # verify the cluster_col provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=cell_cluster_col,
        valid_cluster_cols=['cell_som_cluster', 'cell_meta_cluster']
    )

    # ensure the cluster col will be displayed as an integer and not a float
    cell_data[cell_cluster_col] = cell_data[cell_cluster_col].astype(int)

    # define the file for whole cell
    whole_cell_files = [fov + seg_suffix]

    # load the segmentation labels in for the FOV
    label_map = load_utils.load_imgs_from_dir(
        data_dir=seg_dir, files=whole_cell_files, xr_dim_name='compartments',
        xr_channel_names=['whole_cell'], trim_suffix=seg_suffix.split('.')[0]
    ).loc[fov, ...]

    # use label_cells_by_cluster to create cell masks
    img_data = label_cells_by_cluster(
        fov, cell_data, label_map, fov_col='fov',
        cell_label_column='segmentation_label', cluster_column=cell_cluster_col
    )

    return img_data


def generate_and_save_cell_cluster_masks(fovs: List[str],
                                         base_dir: Union[pathlib.Path, str],
                                         save_dir: Union[pathlib.Path, str],
                                         seg_dir: Union[pathlib.Path, str],
                                         cell_data: pd.DataFrame,
                                         cell_cluster_col: str = 'cell_meta_cluster',
                                         seg_suffix: str = '_whole_cell.tiff',
                                         sub_dir: str = None,
                                         name_suffix: str = ''):
    """Generates cell cluster masks and saves them for downstream analysis.

    Args:
        fovs (List[str]):
            A list of fovs to generate and save pixel masks for.
        base_dir (Union[pathlib.Path, str]):
            The path to the data directory.
        save_dir (Union[pathlib.Path, str]):
            The directory to save the generated cell cluster masks.
        seg_dir (Union[pathlib.Path, str]):
            The path to the segmentation data.
        cell_data (pandas.DataFrame):
            The cell data with both cell SOM and meta cluster assignments
        cell_cluster_col (str, optional):
            Whether to assign SOM or meta clusters. Needs to be `'cell_som_cluster'` or
            `'cell_meta_cluster'`. Defaults to `'cell_meta_cluster'`.
        seg_suffix (str, optional):
            The suffix that the segmentation images use. Defaults to `'_whole_cell.tiff'`.
        sub_dir (str, optional):
            The subdirectory to save the images in. If specified images are saved to
            `"data_dir/sub_dir"`. If `sub_dir = None` the images are saved to `"data_dir"`.
            Defaults to `None`.
        name_suffix (str, optional):
            Specify what to append at the end of every cell mask. Defaults to `''`.
    """

    # create the pixel cluster masks across each fov
    with tqdm(total=len(fovs), desc="Cell Cluster Mask Generation") as cell_mask_progress:
        for fov in fovs:
            # generate the cell mask for the FOV
            cell_mask: np.ndarray =\
                generate_cell_cluster_mask(fov=fov, base_dir=base_dir, seg_dir=seg_dir,
                                           cell_data=cell_data,
                                           cell_cluster_col=cell_cluster_col,
                                           seg_suffix=seg_suffix)

            # save the cell mask generated
            save_fov_mask(fov, data_dir=save_dir, mask_data=cell_mask, sub_dir=sub_dir,
                          name_suffix=name_suffix)

            cell_mask_progress.update(1)


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
        numpy.ndarray:
            The image overlaid with pixel cluster labels
    """

    # path checking
    io_utils.validate_paths([tiff_dir, os.path.join(tiff_dir, chan_file_path),
                             os.path.join(base_dir, pixel_data_dir)])

    # verify the pixel_cluster_col provided is valid
    misc_utils.verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster']
    )

    # verify the fov is valid
    misc_utils.verify_in_list(
        provided_fov_file=[fov + '.feather'],
        consensus_fov_files=os.listdir(os.path.join(base_dir, pixel_data_dir))
    )

    # read the sample channel file to determine size of pixel cluster mask
    channel_data = np.squeeze(io.imread(os.path.join(tiff_dir, chan_file_path)))

    # define an array to hold the overlays for the fov
    # use int16 to allow for Photoshop loading
    img_data = np.zeros((channel_data.shape[0], channel_data.shape[1]), dtype='int16')

    fov_data = feather.read_dataframe(
        os.path.join(base_dir, pixel_data_dir, fov + '.feather')
    )

    # ensure integer display and not float
    fov_data[pixel_cluster_col] = fov_data[pixel_cluster_col].astype(int)

    # get the pixel coordinates
    x_coords = fov_data['row_index'].values
    y_coords = fov_data['column_index'].values

    # convert to 1D indexing
    coordinates = x_coords * img_data.shape[0] + y_coords

    # get the cooresponding cluster labels for each pixel
    cluster_labels = list(fov_data[pixel_cluster_col])

    # assign each coordinate in pixel_cluster_mask to its respective cluster label
    img_subset = img_data.ravel()
    img_subset[coordinates] = cluster_labels
    img_data = img_subset.reshape(img_data.shape)

    return img_data


def generate_and_save_pixel_cluster_masks(fovs: List[str],
                                          base_dir: Union[pathlib.Path, str],
                                          save_dir: Union[pathlib.Path, str],
                                          tiff_dir: Union[pathlib.Path, str],
                                          chan_file: Union[pathlib.Path, str],
                                          pixel_data_dir: Union[pathlib.Path, str],
                                          pixel_cluster_col: str = 'pixel_meta_cluster',
                                          sub_dir: str = None,
                                          name_suffix: str = ''):
    """Generates pixel cluster masks and saves them for downstream analysis.

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
            `"data_dir/sub_dir"`. If `sub_dir = None` the images are saved to `"data_dir"`.
            Defaults to `None`.
        name_suffix (str, optional):
            Specify what to append at the end of every pixel mask. Defaults to `''`.
    """

    # create the pixel cluster masks across each fov
    with tqdm(total=len(fovs), desc="Pixel Cluster Mask Generation") as pixel_mask_progress:
        for fov in fovs:
            # define the path to provided channel file in the fov dir, used to calculate dimensions
            chan_file_path = os.path.join(fov, chan_file)

            # generate the pixel mask for the FOV
            pixel_mask: np.ndarray =\
                generate_pixel_cluster_mask(fov=fov, base_dir=base_dir, tiff_dir=tiff_dir,
                                            chan_file_path=chan_file_path,
                                            pixel_data_dir=pixel_data_dir,
                                            pixel_cluster_col=pixel_cluster_col)

            # save the pixel mask generated
            save_fov_mask(fov, data_dir=save_dir, mask_data=pixel_mask, sub_dir=sub_dir,
                          name_suffix=name_suffix)

            pixel_mask_progress.update(1)


def generate_and_save_neighborhood_cluster_masks(fovs: List[str],
                                                 save_dir: Union[pathlib.Path, str],
                                                 neighborhood_data: pd.DataFrame,
                                                 seg_dir: str,
                                                 seg_suffix: str = '_whole_cell.tiff',
                                                 xr_channel_name='segmentation_label',
                                                 sub_dir: str = None,
                                                 name_suffix: str = ''):
    """Generates neighborhood cluster masks and saves them for downstream analysis

    Args:
        fovs (List[str]):
            A list of fovs to generate and save neighborhood masks for.
        save_dir (Union[pathlib.Path, str]):
            The directory to save the generated pixel cluster masks.
        neighborhood_data (pandas.DataFrame):
            Contains the neighborhood cluster assignments for each cell.
        seg_dir (str):
            The path to the segmentation data.
        seg_suffix (str):
            The suffix that the segmentation images use. Defaults to `'_whole_cell.tiff'`.
        xr_channel_name (str):
            Channel name for segmented data array.
        sub_dir (str, optional):
            The subdirectory to save the images in. If specified images are saved to
            `"data_dir/sub_dir"`. If `sub_dir = None` the images are saved to `"data_dir"`.
            Defaults to `None`.
        name_suffix (str, optional):
            Specify what to append at the end of every pixel mask. Defaults to `''`.
    """

    # create the neighborhood cluster masks across each fov
    with tqdm(total=len(fovs), desc="Neighborhood Cluster Mask Generation") as neigh_mask_progress:
        # generate the mask for each FOV
        for fov in fovs:
            # load in the label map for the FOV
            label_map = load_utils.load_imgs_from_dir(
                seg_dir, files=[fov + seg_suffix], xr_channel_names=[xr_channel_name],
                trim_suffix=seg_suffix.split('.')[0]
            ).loc[fov, ..., :]

            # generate the neighborhood mask for the FOV
            neighborhood_mask: np.ndarray =\
                label_cells_by_cluster(
                    fov, neighborhood_data, label_map
                )

            # save the neighborhood mask generated
            save_fov_mask(fov, data_dir=save_dir, mask_data=neighborhood_mask, sub_dir=sub_dir,
                          name_suffix=name_suffix)

            neigh_mask_progress.update(1)


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
            image_utils.save_image(save_path, channel)


def stitch_images_by_shape(data_dir, stitched_dir, img_sub_folder=None, channels=None,
                           segmentation=False, clustering=False):
    """ Creates stitched images for the specified channels based on the FOV folder names

    Args:
        data_dir (str):
            path to directory containing images
        stitched_dir (str):
            path to directory to save stitched images to
        img_sub_folder (str):
            optional name of image sub-folder within each fov
        channels (list):
            optional list of imgs to load, otherwise loads all imgs
        segmentation (bool):
            if stitching images from the single segmentation dir
        clustering (bool or str):
            if stitching images from the single pixel or cell mask dir, specify 'pixel' / 'cell'
    """

    io_utils.validate_paths(data_dir)

    # no img_sub_folder, change to empty string to read directly from base folder
    if img_sub_folder is None:
        img_sub_folder = ""

    if clustering and clustering not in ['pixel', 'cell']:
        raise ValueError('If stitching images from the pixie pipeline, the clustering arg must be '
                         'set to either \"pixel\" or \"cell\".')

    # retrieve valid fov names
    if segmentation:
        fovs = ns.natsorted(io_utils.list_files(data_dir, substrs='_whole_cell.tiff'))
        fovs = io_utils.extract_delimited_names(fovs, delimiter='_whole_cell.tiff')
    elif clustering:
        fovs = ns.natsorted(io_utils.list_files(data_dir, substrs=f'_{clustering}_mask.tiff'))
        fovs = io_utils.extract_delimited_names(fovs, delimiter=f'_{clustering}_mask.tiff')
    else:
        fovs = ns.natsorted(io_utils.list_folders(data_dir))
        # ignore previous toffy stitching in fov directory
        if 'stitched_images' in fovs:
            fovs.remove('stitched_images')

    if len(fovs) == 0:
        raise ValueError(f"No FOVs found in directory, {data_dir}.")

    # check previous stitching
    if os.path.exists(stitched_dir):
        raise ValueError(f"The {stitched_dir} directory already exists.")

    bad_fov_names = []
    for fov in fovs:
        r = re.compile('.*R.*C.*')
        if r.match(fov) is None:
            bad_fov_names.append(fov)
    if len(bad_fov_names) > 0:
        raise ValueError(f"Invalid FOVs found in directory, {data_dir}. FOV names "
                         f"{bad_fov_names} should have the form RnCm.")

    # retrieve all extracted channel names and verify list if provided
    if not segmentation and not clustering:
        channel_imgs = io_utils.list_files(
            dir_name=os.path.join(data_dir, fovs[0], img_sub_folder),
            substrs=['.tiff', '.jpg', '.png'])
    else:
        channel_imgs = io_utils.list_files(data_dir, substrs=fovs[0])
        channel_imgs = [chan.split(fovs[0] + '_')[1] for chan in channel_imgs]

    if channels is None:
        channels = io_utils.remove_file_extensions(channel_imgs)
    else:
        misc_utils.verify_in_list(channel_inputs=channels,
                                  valid_channels=io_utils.remove_file_extensions(channel_imgs))

    os.makedirs(stitched_dir)

    file_ext = channel_imgs[0].split('.')[1]
    expected_fovs, num_rows, num_cols = load_utils.get_tiled_fov_names(fovs, return_dims=True)

    # save new images to the stitched_images, one channel at a time
    for chan in channels:
        image_data = load_utils.load_tiled_img_data(data_dir, fovs, expected_fovs, chan,
                                                    single_dir=any([segmentation, clustering]),
                                                    file_ext=file_ext,
                                                    img_sub_folder=img_sub_folder)
        stitched_data = data_utils.stitch_images(image_data, num_cols)
        current_img = stitched_data.loc['stitched_image', :, :, chan].values
        image_utils.save_image(os.path.join(stitched_dir, chan + '_stitched.' + file_ext),
                               current_img)
