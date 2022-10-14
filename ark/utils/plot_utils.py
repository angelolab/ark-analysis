import os
import pathlib
import shutil
from operator import contains
from typing import List, Union

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries

from ark.utils import io_utils, load_utils, misc_utils
# plotting functions
from ark.utils.misc_utils import verify_in_list, verify_same_elements


def plot_neighborhood_cluster_result(img_xr, fovs, k, save_dir=None, cmap_name='tab20',
                                     fov_col='fovs', figsize=(10, 10)):
    """Takes an xarray containing labeled images and displays them.
    Args:
        img_xr (xarray.DataArray):
            xarray containing labeled cell objects.
        fovs (list):
            list of fovs to display.
        k (int):
            number of clusters (neighborhoods)
        save_dir (str):
            If provided, the image will be saved to this location.
        cmap_name (str):
            Cmap to use for the image that will be displayed.
        fov_col (str):
            column with the fovs names in `img_xr`.
        figsize (tuple):
            Size of the image that will be displayed.
    """

    # verify the fovs are valid
    verify_in_list(fov_names=fovs, unique_fovs=img_xr.fovs.values)

    # define the colormap, add black for empty slide
    mycols = cm.get_cmap(cmap_name, k).colors
    mycols = np.vstack(([0, 0, 0, 1], mycols))
    cmap = colors.ListedColormap(mycols)
    bounds = [i-0.5 for i in np.linspace(0, k+1, k+2)]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    for fov in fovs:
        # define the figure
        plt.figure(figsize=figsize)

        # define the axis
        ax = plt.gca()

        # make the title
        plt.title(fov)

        # show the image on the figure
        im = plt.imshow(img_xr[img_xr[fov_col] == fov].values.squeeze(),
                        cmap=cmap, norm=norm)

        # remove the axes
        plt.axis('off')

        # remove the gridlines
        plt.grid(visible=None)

        # ensure the colorbar matches up with the margins on the right
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # draw the colorbar
        tick_names = ['Cluster'+str(x) for x in range(1, k+1)]
        tick_names = ['Empty'] + tick_names
        cbar = plt.colorbar(im, cax=cax, ticks=np.arange(len(tick_names)))
        cbar.set_ticks(cbar.ax.get_yticks())
        cbar.ax.set_yticklabels(tick_names)

        # save if specified
        if save_dir:
            misc_utils.save_figure(save_dir, f'{fov}.png')


# TODO: possibly need to merge this with plot_neighborhood_cluster_result
def plot_pixel_cell_cluster_overlay(img_xr, fovs, cluster_id_to_name_path, metacluster_colors,
                                    save_dir=None, fov_col='fovs', figsize=(10, 10)):
    """Overlays the pixel and cell clusters on an image

    Args:
        img_xr (xarray.DataArray):
            xarray containing labeled pixel or cell clusters
        fovs (list):
            list of fovs to display
        cluster_id_to_name_path (str):
            a path to a CSV identifying the pixel/cell cluster to manually-defined name mapping
            this is output by the remapping visualization found in `metacluster_remap_gui`
        metacluster_colors (dict):
            maps each metacluster id to a color
        save_dir (str):
            If provided, the image will be saved to this location.
        fov_col (str):
            column with the fovs names in `img_xr`.
        figsize (tuple):
            Size of the image that will be displayed.
    """

    # verify the fovs are valid
    verify_in_list(fov_names=fovs, unique_fovs=img_xr.fovs.values)

    # verify cluster_id_to_name_path exists
    if not os.path.exists(cluster_id_to_name_path):
        raise FileNotFoundError(
            'Metacluster id to renamed metacluster mapping %s does not exist' %
            cluster_id_to_name_path
        )

    # read the cluster to name mapping
    cluster_id_to_name = pd.read_csv(cluster_id_to_name_path)

    # this mapping file needs to contain the following columns:
    # 'cluster', 'metacluster', and 'mc_name'
    # NOTE: check for 'cluster' ensures this file was generated by the interactive visualization
    verify_same_elements(
        cluster_mapping_cols=cluster_id_to_name.columns.values,
        required_cols=['cluster', 'metacluster', 'mc_name']
    )

    # subset on just metacluster and mc_name
    metacluster_id_to_name = cluster_id_to_name[['metacluster', 'mc_name']].copy()

    # take only the unique pairs
    metacluster_id_to_name = metacluster_id_to_name.drop_duplicates()

    # sort by metacluster id ascending, this will help when making the colormap
    metacluster_id_to_name = metacluster_id_to_name.sort_values(by='metacluster')

    # assert the metacluster index in colors matches with the ids in metacluster_id_to_name
    verify_same_elements(
        metacluster_colors_ids=list(metacluster_colors.keys()),
        metacluster_mapping_ids=metacluster_id_to_name['metacluster'].values
    )

    # use metacluster_colors to add the colors to metacluster_id_to_name
    metacluster_id_to_name['color'] = metacluster_id_to_name['metacluster'].map(
        metacluster_colors
    )

    # need to add black to denote a pixel with no clusters
    mc_colors = [(0.0, 0.0, 0.0)] + list(metacluster_id_to_name['color'].values)

    # map each metacluster_id_to_name to its index + 1
    # NOTE: explicitly needed to ensure correct colormap colors are drawn and colorbar
    # is indexed correctly when plotted
    metacluster_to_index = {}
    for index, row in metacluster_id_to_name.reset_index(drop=True).iterrows():
        metacluster_to_index[row['metacluster']] = index + 1

    # generate the colormap
    cmap = colors.ListedColormap(mc_colors)
    norm = colors.BoundaryNorm(
        np.linspace(0, len(mc_colors), len(mc_colors) + 1) - 0.5,
        len(mc_colors)
    )

    for fov in fovs:
        # retrieve the image associated with the FOV
        fov_img = img_xr[img_xr[fov_col] == fov].values

        # assign any metacluster id not in metacluster_id_to_name to 0 (not including 0 itself)
        # done as a precaution, should not usually happen
        acceptable_cluster_ids = [0] + list(metacluster_id_to_name['metacluster'])
        fov_img[~np.isin(fov_img, acceptable_cluster_ids)] = 0

        # explicitly relabel each value in fov_img with its index in mc_colors
        # to ensure proper indexing into colormap
        for mc, mc_index in metacluster_to_index.items():
            fov_img[fov_img == mc] = mc_index

        # define the figure
        fig = plt.figure(figsize=figsize)

        # make the title
        plt.title(fov)

        # display the image
        overlay = plt.imshow(
            fov_img.squeeze(),
            cmap=cmap,
            norm=norm,
            origin='upper'
        )

        # remove the axes
        plt.axis('off')

        # remove the gridlines
        plt.grid(b=None)

        # define the colorbar with annotations
        cax = fig.add_axes([0.9, 0.1, 0.01, 0.8])
        cbar = plt.colorbar(
            overlay,
            ticks=np.arange(len(mc_colors)),
            cax=cax,
            orientation='vertical'
        )
        cbar.ax.set_yticklabels(['Empty'] + list(metacluster_id_to_name['mc_name'].values))

        # save if specified
        if save_dir:
            misc_utils.save_figure(save_dir, f'{fov}.png')


def tif_overlay_preprocess(segmentation_labels, plotting_tif):
    """Validates plotting_tif and preprocesses it accordingly
    Args:
        segmentation_labels (numpy.ndarray):
            2D numpy array of labeled cell objects
        plotting_tif (numpy.ndarray):
            2D or 3D numpy array of imaging signal
    Returns:
        numpy.ndarray:
            The preprocessed image
    """

    if len(plotting_tif.shape) == 2:
        if plotting_tif.shape != segmentation_labels.shape:
            raise ValueError("plotting_tif and segmentation_labels array dimensions not equal.")
        else:
            # convert RGB image with the blue channel containing the plotting tif data
            formatted_tif = np.zeros((plotting_tif.shape[0], plotting_tif.shape[1], 3),
                                     dtype=plotting_tif.dtype)
            formatted_tif[..., 2] = plotting_tif
    elif len(plotting_tif.shape) == 3:
        # can only support up to 3 channels
        if plotting_tif.shape[2] > 3:
            raise ValueError("max 3 channels of overlay supported, got {}".
                             format(plotting_tif.shape))

        # set first n channels (in reverse order) of formatted_tif to plotting_tif
        # (n = num channels in plotting_tif)
        formatted_tif = np.zeros((plotting_tif.shape[0], plotting_tif.shape[1], 3),
                                 dtype=plotting_tif.dtype)
        formatted_tif[..., :plotting_tif.shape[2]] = plotting_tif
        formatted_tif = np.flip(formatted_tif, axis=2)
    else:
        raise ValueError("plotting tif must be 2D or 3D array, got {}".
                         format(plotting_tif.shape))

    return formatted_tif


def create_overlay(fov, segmentation_dir, data_dir,
                   img_overlay_chans, seg_overlay_comp, alternate_segmentation=None):
    """Take in labeled contour data, along with optional mibi tif and second contour,
    and overlay them for comparison"
    Generates the outline(s) of the mask(s) as well as intensity from plotting tif. Predicted
    contours are colored red, while alternate contours are colored white.

    Args:
        fov (str):
            The name of the fov to overlay
        segmentation_dir (str):
            The path to the directory containing the segmentatation data
        data_dir (str):
            The path to the directory containing the nuclear and whole cell image data
        img_overlay_chans (list):
            List of channels the user will overlay
        seg_overlay_comp (str):
            The segmentted compartment the user will overlay
        alternate_segmentation (numpy.ndarray):
            2D numpy array of labeled cell objects
    Returns:
        numpy.ndarray:
            The image with the channel overlay
    """

    # load the specified fov data in
    plotting_tif = load_utils.load_imgs_from_dir(
        data_dir=data_dir,
        files=[fov + '.tif'],
        xr_dim_name='channels',
        xr_channel_names=['nuclear_channel', 'membrane_channel']
    )

    # verify that the provided image channels exist in plotting_tif
    misc_utils.verify_in_list(
        provided_channels=img_overlay_chans,
        img_channels=plotting_tif.channels.values
    )

    # subset the plotting tif with the provided image overlay channels
    plotting_tif = plotting_tif.loc[fov, :, :, img_overlay_chans].values

    # read the segmentation data in
    segmentation_labels_cell = load_utils.load_imgs_from_dir(data_dir=segmentation_dir,
                                                             files=[fov + '_feature_0.tif'],
                                                             xr_dim_name='compartments',
                                                             xr_channel_names=['whole_cell'],
                                                             trim_suffix='_feature_0',
                                                             match_substring='_feature_0')
    segmentation_labels_nuc = load_utils.load_imgs_from_dir(data_dir=segmentation_dir,
                                                            files=[fov + '_feature_1.tif'],
                                                            xr_dim_name='compartments',
                                                            xr_channel_names=['nuclear'],
                                                            trim_suffix='_feature_1',
                                                            match_substring='_feature_1')

    segmentation_labels = xr.DataArray(np.concatenate((segmentation_labels_cell.values,
                                                      segmentation_labels_nuc.values),
                                                      axis=-1),
                                       coords=[segmentation_labels_cell.fovs,
                                               segmentation_labels_cell.rows,
                                               segmentation_labels_cell.cols,
                                               ['whole_cell', 'nuclear']],
                                       dims=segmentation_labels_cell.dims)

    # verify that the provided segmentation channels exist in segmentation_labels
    misc_utils.verify_in_list(
        provided_compartments=seg_overlay_comp,
        seg_compartments=segmentation_labels.compartments.values
    )

    # subset segmentation labels with the provided segmentation overlay channels
    segmentation_labels = segmentation_labels.loc[fov, :, :, seg_overlay_comp].values

    # overlay the segmentation labels over the image
    plotting_tif = tif_overlay_preprocess(segmentation_labels, plotting_tif)

    # define borders of cells in mask
    predicted_contour_mask = find_boundaries(segmentation_labels,
                                             connectivity=1, mode='inner').astype(np.uint8)
    predicted_contour_mask[predicted_contour_mask > 0] = 255

    # rescale each channel to go from 0 to 255
    rescaled = np.zeros(plotting_tif.shape, dtype='uint8')

    for idx in range(plotting_tif.shape[2]):
        if np.max(plotting_tif[:, :, idx]) == 0:
            # don't need to rescale this channel
            pass
        else:
            percentiles = np.percentile(plotting_tif[:, :, idx][plotting_tif[:, :, idx] > 0],
                                        [5, 95])
            rescaled_intensity = rescale_intensity(plotting_tif[:, :, idx],
                                                   in_range=(percentiles[0], percentiles[1]),
                                                   out_range='uint8')
            rescaled[:, :, idx] = rescaled_intensity

    # overlay first contour on all three RGB, to have it show up as white border
    rescaled[predicted_contour_mask > 0, :] = 255

    # overlay second contour as red outline if present
    if alternate_segmentation is not None:

        if segmentation_labels.shape != alternate_segmentation.shape:
            raise ValueError("segmentation_labels and alternate_"
                             "segmentation array dimensions not equal.")

        # define borders of cell in mask
        alternate_contour_mask = find_boundaries(alternate_segmentation, connectivity=1,
                                                 mode='inner').astype(np.uint8)
        rescaled[alternate_contour_mask > 0, 0] = 255
        rescaled[alternate_contour_mask > 0, 1:] = 0

    return rescaled


def set_minimum_color_for_colormap(cmap, default=(0, 0, 0, 1)):
    """ Changes minimum value in provided colormap to black (#000000) or provided color

    This is useful for instances where zero-valued regions of an image should be
    distinct from positive regions (i.e transparent or non-colormap member color)

    Args:
        cmap (matplotlib.colors.Colormap):
            matplotlib color map
        default (Iterable):
            RGBA color values for minimum color. Default is black, (0, 0, 0, 1).

    Returns:
        matplotlib.colors.Colormap:
            corrected colormap
    """
    cmapN = cmap.N
    corrected = cmap(np.arange(cmapN))
    corrected[0, :] = list(default)
    return colors.ListedColormap(corrected)


def create_mantis_dir(fovs: List[str], mantis_project_path: Union[str, pathlib.Path],
                      img_data_path: Union[str, pathlib.Path],
                      mask_output_dir: Union[str, pathlib.Path],
                      mapping: Union[str, pathlib.Path, pd.DataFrame],
                      seg_dir: Union[str, pathlib.Path],
                      mask_suffix: str = "_mask", img_sub_folder: str = ""):
    """Creates a mantis project directory so that it can be opened by the mantis viewer.
    Copies fovs, segmentation files, masks, and mapping csv's into a new directory structure.
    Here is how the contents of the mantis project folder will look like.

    ```{code-block} sh
    mantis_project
    ├── fov0
    │   ├── cell_segmentation.tiff
    │   ├── chan0.tiff
    │   ├── chan1.tiff
    │   ├── chan2.tiff
    │   ├── ...
    │   ├── population_mask.csv
    │   └── population_mask.tiff
    └── fov1
    │   ├── cell_segmentation.tiff
    │   ├── chan0.tiff
    │   ├── chan1.tiff
    │   ├── chan2.tiff
    │   ├── ...
    │   ├── population_mask.csv
    │   └── population_mask.tiff
    └── ...
    ```

    Args:
        fovs (List[str]):
            A list of FOVs to create a Mantis Project for.
        mantis_project_path (Union[str, pathlib.Path]):
            The folder where the mantis project will be created.
        img_data_path (Union[str, pathlib.Path]):
            The location of the all the fovs you wish to create a project from.
        mask_output_dir (Union[str, pathlib.Path]):
            The folder containing all the masks of the fovs.
        mapping (Union[str, pathlib.Path, pd.DataFrame]):
            The location of the mapping file, or the mapping Pandas DataFrame itself.
        seg_dir (Union[str, pathlib.Path]):
            The location of the segmentation directory for the fovs.
        mask_suffix (str, optional):
            The suffix used to find the mask tiffs. Defaults to "_mask".
        img_sub_folder (str, optional):
            The subfolder where the channels exist within the `img_data_path`.
            Defaults to "normalized".
    """

    if not os.path.exists(mantis_project_path):
        os.makedirs(mantis_project_path)

    # create key from cluster number to cluster name
    if type(mapping) in {pathlib.Path, str}:
        map_df = pd.read_csv(mapping)
    elif type(mapping) is pd.DataFrame:
        map_df = mapping
    else:
        ValueError("Mapping must either be a path to an already saved mapping csv, \
                   or a DataFrame that is already loaded in.")

    map_df = map_df.loc[:, ['metacluster', 'mc_name']]
    # remove duplicates from df
    map_df = map_df.drop_duplicates()
    map_df = map_df.sort_values(by=['metacluster'])

    # rename for mantis names
    map_df = map_df.rename({'metacluster': 'region_id', 'mc_name': 'region_name'}, axis=1)

    # get names of fovs with masks
    mask_names_loaded = (io_utils.list_files(mask_output_dir, mask_suffix))
    mask_names_delimited = io_utils.extract_delimited_names(mask_names_loaded,
                                                            delimiter=mask_suffix)
    mask_names_sorted = natsort.natsorted(mask_names_delimited)

    # use `fovs`, a subset of the FOVs in `total_fov_names` which
    # is a list of FOVs in `img_data_path`
    fovs = natsort.natsorted(fovs)
    verify_in_list(fovs=fovs, img_data_fovs=mask_names_delimited)

    # Filter out the masks that do not have an associated FOV.
    mask_names = filter(lambda mn: any(contains(mn, f) for f in fovs), mask_names_sorted)

    # create a folder with image data, pixel masks, and segmentation mask
    for fov, mn in zip(fovs, mask_names):
        # set up paths
        img_source_dir = os.path.join(img_data_path, fov, img_sub_folder)
        output_dir = os.path.join(mantis_project_path, fov)

        # copy image data if not already copied in from previous round of clustering
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

            # copy all channels into new folder
            chans = io_utils.list_files(img_source_dir, '.tiff')
            for chan in chans:
                shutil.copy(os.path.join(img_source_dir, chan), os.path.join(output_dir, chan))

        # copy mask into new folder
        mask_name = mn + mask_suffix + '.tiff'
        shutil.copy(os.path.join(mask_output_dir, mask_name),
                    os.path.join(output_dir, 'population{}.tiff'.format(mask_suffix)))

        # copy the segmentation files into the output directory
        seg_name = fov + '_feature_0.tif'
        shutil.copy(os.path.join(seg_dir, seg_name),
                    os.path.join(output_dir, 'cell_segmentation.tiff'))

        # copy mapping into directory
        map_df.to_csv(os.path.join(output_dir, 'population{}.csv'.format(mask_suffix)),
                      index=False)
