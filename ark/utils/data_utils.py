import math
import os
import pathlib
import shutil
from typing import List, Union

import re
import datasets
import feather
import numpy as np
import skimage.io as io
import xarray as xr
import natsort as ns
from tqdm.notebook import tqdm_notebook as tqdm

from ark import settings
from ark.utils import load_utils, io_utils
from ark.utils.load_utils import load_tiled_img_data
from ark.utils.misc_utils import verify_in_list


def save_fov_images(fovs, data_dir, img_xr, sub_dir=None, name_suffix=''):
    """Given an xarray of images per fov, saves each image separately

    Args:
        fovs (list):
            List of fovs to save in img_xr
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

    if sub_dir is not None:
        # Save the fovs in the directory `data_dir/sub_dir/`
        save_dir = os.path.join(data_dir, sub_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        # Save the fovs in the directory `data_dir`
        save_dir = data_dir

    for fov in fovs:
        # retrieve the image for the fov
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


def generate_cell_cluster_mask(fovs, base_dir, seg_dir, cell_data_name,
                               cell_cluster_col='cell_meta_cluster', seg_suffix='_feature_0.tif'):
    """For each fov, create a mask labeling each cell with their SOM or meta cluster label

    Args:
        fovs (list):
            List of fovs to relabel
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

    # verify all the fovs are valid
    verify_in_list(
        provided_fovs=fovs,
        consensus_fovs=cell_consensus_data['fov']
    )

    # define the files for whole cell and nuclear
    whole_cell_files = [fov + seg_suffix for fov in fovs]

    # load the segmentation labels in
    label_maps = load_utils.load_imgs_from_dir(data_dir=seg_dir,
                                               files=whole_cell_files,
                                               xr_dim_name='compartments',
                                               xr_channel_names=['whole_cell'],
                                               trim_suffix=seg_suffix.split('.')[0])
    # use label_cells_by_cluster to create cell masks
    img_data = label_cells_by_cluster(
        fovs, cell_consensus_data, label_maps, fov_col='fov',
        cell_label_column='segmentation_label', cluster_column=cell_cluster_col
    )

    return img_data


def generate_pixel_cluster_mask(fovs, base_dir, tiff_dir, chan_file,
                                pixel_data_dir, pixel_cluster_col='pixel_meta_cluster'):
    """For each fov, create a mask labeling each pixel with their SOM or meta cluster label

    Args:
        fovs (list):
            List of fovs to relabel
        base_dir (str):
            The path to the data directory
        tiff_dir (str):
            The path to the tiff data
        chan_file (str):
            The path to the sample channel file to load (assuming `tiff_dir` as root)
            Only used to determine dimensions of the pixel mask.
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

    if not os.path.exists(os.path.join(tiff_dir, chan_file)):
        raise FileNotFoundError("chan_file %s does not exist in tiff_dir %s"
                                % (chan_file, tiff_dir))

    if not os.path.exists(os.path.join(base_dir, pixel_data_dir)):
        raise FileNotFoundError(
            "Pixel data dir %s does not exist in base_dir %s" % (pixel_data_dir, base_dir)
        )

    # verify the pixel_cluster_col provided is valid
    verify_in_list(
        provided_cluster_col=[pixel_cluster_col],
        valid_cluster_cols=['pixel_som_cluster', 'pixel_meta_cluster']
    )

    # verify all the fovs are valid
    verify_in_list(
        provided_fov_files=[fov + '.feather' for fov in fovs],
        consensus_fov_files=os.listdir(os.path.join(base_dir, pixel_data_dir))
    )

    # read the sample channel file to determine size of pixel cluster mask
    channel_data = np.squeeze(io.imread(os.path.join(tiff_dir, chan_file)))

    # define an array to hold the overlays for each fov
    img_data = np.zeros((len(fovs), channel_data.shape[0], channel_data.shape[1]))

    for i, fov in enumerate(fovs):
        # read the pixel data for the fov
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
        img_subset = img_data[i, ...].ravel()
        img_subset[coordinates] = cluster_labels
        img_data[i, ...] = img_subset.reshape(img_data[i, ...].shape)

    # create the stacked img_data xarray and return
    return xr.DataArray(img_data, coords=[fovs, range(img_data[0].shape[0]),
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
                                          name_suffix: str = '',
                                          batch_size=5):
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
            The path to the sample channel file to load (assuming `tiff_dir` as root)
            Only used to determine dimensions of the pixel mask.
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
        batch_size (int, optional):
            The number of fovs to process at once for each batch. Defaults to 5.
    """

    # define a list of fov batches to process over
    fov_batches = [fovs[i:i + batch_size] for i in range(0, len(fovs), batch_size)]

    # create the pixel cluster masks over each fov batch.
    with tqdm(total=len(fovs), desc="Pixel Cluster Mask Generation") as pixel_mask_progress:
        for fov_batch in fov_batches:
            pixel_masks: xr.DataArray =\
                generate_pixel_cluster_mask(fovs=fov_batch, base_dir=base_dir, tiff_dir=tiff_dir,
                                            chan_file=chan_file, pixel_data_dir=pixel_data_dir,
                                            pixel_cluster_col=pixel_cluster_col)

            save_fov_images(fov_batch, data_dir=save_dir, img_xr=pixel_masks, sub_dir=sub_dir,
                            name_suffix=name_suffix)

            pixel_mask_progress.update(len(fov_batch))


def generate_and_save_cell_cluster_masks(fovs: List[str],
                                         base_dir: Union[pathlib.Path, str],
                                         save_dir: Union[pathlib.Path, str],
                                         seg_dir: Union[pathlib.Path, str],
                                         cell_data_name: Union[pathlib.Path, str],
                                         cell_cluster_col: str = 'cell_meta_cluster',
                                         seg_suffix: str = '_feature_0.tif',
                                         sub_dir: str = None,
                                         name_suffix: str = '',
                                         batch_size=5):
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
        batch_size (int, optional):
            The number of fovs to process at once for each batch. Defaults to 5.
    """

    # define a list of fov batches to process over
    fov_batches = [fovs[i:i + batch_size] for i in range(0, len(fovs), batch_size)]

    # create the pixel cluster masks over each fov batch.
    with tqdm(total=len(fovs), desc="Cell Cluster Mask Generation") as cell_mask_progress:
        for fov_batch in fov_batches:
            cell_masks: xr.DataArray =\
                generate_cell_cluster_mask(fovs=fov_batch, base_dir=base_dir, seg_dir=seg_dir,
                                           cell_data_name=cell_data_name,
                                           cell_cluster_col=cell_cluster_col,
                                           seg_suffix=seg_suffix)

            save_fov_images(fov_batch, data_dir=save_dir, img_xr=cell_masks, sub_dir=sub_dir,
                            name_suffix=name_suffix)

            cell_mask_progress.update(len(fov_batch))


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
                            is_mibitiff=False, img_sub_folder="TIFs", dtype="int16"):
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

    for fov in fovs:
        # load the images in the current fov batch
        if is_mibitiff:
            data_xr = load_utils.load_imgs_from_mibitiff(
                tiff_dir, mibitiff_files=[fov], channels=channels
            )
        else:
            data_xr = load_utils.load_imgs_from_tree(
                tiff_dir, img_sub_folder=img_sub_folder, fovs=[fov], channels=channels
            )

        fov_name = data_xr.fovs.values[0]
        out = np.zeros((2, data_xr.shape[1], data_xr.shape[2]), dtype=data_xr.dtype)

        # sum over channels and add to output
        if nuc_channels:
            out[0] = np.sum(data_xr.loc[fov_name, :, :, nuc_channels].values, axis=2)
        if mem_channels:
            out[1] = np.sum(data_xr.loc[fov_name, :, :, mem_channels].values, axis=2)

        save_path = os.path.join(data_dir, f"{fov_name}.tif")
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

    data_path = pathlib.Path(ds["base_dataset"]["Data Path"][0]) / "input_data"

    shutil.copytree(data_path, pathlib.Path(save_dir) / "image_data",
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('._*'))


def stitch_images_by_shape(data_dir, stitched_dir, img_sub_folder=None, channels=None,
                           max_img_size=None, segmentation_dir=False, pixie_dir=False):
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
        max_img_size (int or None):
            The length (in pixels) of the largest image that will be loaded. All other images will
            be padded to bring them up to the same size.
        segmentation_dir (bool):
            if stitching images from the single segemenation dir
        pixie_dir (bool / str):
            if stitching images from the single pixel or cell mask dir, specify 'pixel' / 'cell'
    """

    # no img_sub_folder, change to empty string to read directly from base folder
    if img_sub_folder is None:
        img_sub_folder = ""

    if pixie_dir and pixie_dir not in ['pixel', 'cell']:
        raise ValueError('If stitching images from the pixie pipeline, the pixie_dir arg must be '
                         'set to either \"pixel\" or \"cell\".')

    # retrieve valid fov names
    if segmentation_dir:
        fovs = ns.natsorted(io_utils.list_files(data_dir, substrs='_feature_0.tif'))
        fovs = io_utils.extract_delimited_names(fovs, delimiter='_feature_0.tif')
    elif pixie_dir:
        fovs = ns.natsorted(io_utils.list_files(data_dir, substrs=f'_{pixie_dir}_mask.tif'))
        fovs = io_utils.extract_delimited_names(fovs, delimiter=f'_{pixie_dir}_mask.tif')
    else:
        fovs = ns.natsorted(io_utils.list_folders(data_dir))
        if 'stitched_images' in fovs:
            fovs.remove('stitched_images')

    if len(fovs) == 0:
        raise ValueError(f"No FOVs found in directory, {data_dir}.")

    # check previous tiling
    if os.path.exists(stitched_dir):
        raise ValueError(f"The {stitched_dir} directory already exists.")

    bad_dirs = []
    for fov in fovs:
        r = re.compile('.*R.*C.*')
        if r.match(fov) is None:
            bad_dirs.append(fov)
    if len(bad_dirs) > 0:
        raise ValueError(f"Invalid FOVs found in directory, {data_dir}. FOV names "
                         f"{bad_dirs} should have the form RnCm.")

    # retrieve all extracted channel names, or verify the list provided
    if not segmentation_dir and not pixie_dir:
        channel_imgs = io_utils.list_files(
            dir_name=os.path.join(data_dir, fovs[0], img_sub_folder),
            substrs=['.tif', '.jpg', '.png'])
    else:
        channel_imgs = io_utils.list_files(data_dir, substrs=fovs[0])
        channel_imgs = [chan.split(fovs[0] + '_')[1] for chan in channel_imgs]

    if channels is None:
        channels = io_utils.remove_file_extensions(channel_imgs)
    else:
        verify_in_list(channel_inputs=channels,
                       valid_channels=io_utils.remove_file_extensions(channel_imgs))

    # make stitched subdir
    os.makedirs(stitched_dir)
    if max_img_size is None:
        max_img_size = load_utils.get_max_img_size(data_dir, img_sub_folder,
                                                   single_dir=any([segmentation_dir, pixie_dir]))

    file_ext = channel_imgs[0].split('.')[1]
    _, num_rows, num_cols = load_utils.get_tiled_fov_names(fovs, return_dims=True)

    # save the stitched images to the stitched_images subdir, one channel at a time
    for chan in channels:
        image_data = load_tiled_img_data(data_dir, fovs, chan,
                                         single_dir=any([segmentation_dir, pixie_dir]),
                                         max_image_size=max_img_size,
                                         file_ext=file_ext, img_sub_folder=img_sub_folder)
        stitched_data = stitch_images(image_data, num_cols)
        current_img = stitched_data.loc['stitched_image', :, :, chan].values
        io.imsave(os.path.join(stitched_dir, chan + '_stitched.' + file_ext),
                  current_img.astype('float32'), check_contrast=False)
