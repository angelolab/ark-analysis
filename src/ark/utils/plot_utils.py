import os
import pathlib
import shutil
from dataclasses import dataclass, field
from operator import contains
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import xarray as xr
from alpineer import image_utils, io_utils, load_utils, misc_utils
from alpineer.settings import EXTENSION_TYPES
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.exposure import rescale_intensity
from skimage.segmentation import find_boundaries
from tqdm import tqdm


@dataclass
class MetaclusterColormap:
    """
    A dataclass which contains the colormap-related information for the metaclusters.

    """
    cluster_type: str
    cluster_id_to_name_path: Union[str, pathlib.Path]
    metacluster_colors: Dict

    # Fields initialized after `__post_init__`
    unassigned_color: Tuple[float, ...] = field(init=False)
    unassigned_id: int = field(init=False)
    no_cluster_color: Tuple[float, ...] = field(init=False)
    metacluster_id_to_name: pd.DataFrame = field(init=False)
    mc_colors: np.ndarray = field(init=False)
    metacluster_to_index: Dict = field(init=False)
    cmap: colors.ListedColormap = field(init=False)
    norm: colors.BoundaryNorm = field(init=False)

    def __post_init__(self) -> None:
        """
        Initializes the fields of the dataclass after the object is instantiated.
        """

        # A pixel with no associated metacluster (gray, #5A5A5A)
        self.unassigned_color: Tuple[float, ...] = (0.9, 0.9, 0.9, 1.0)

        # A pixel assigned to no cluster (black, #000000)
        self.no_cluster_color: Tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)

        self._metacluster_cmap_generator()

    def _metacluster_cmap_generator(self) -> None:
        """
        A helper function which generates a colormap for the metaclusters with a given cluster ID
        to name mapping.

        Args:
            cluster_id_to_name_path (Union[str, pathlib.Path]):
                a path to a CSV identifying the pixel/cell cluster to manually-defined name mapping
                this is output by the remapping visualization found in `metacluster_remap_gui`
            metacluster_colors (Dict):
                maps each metacluster id to a color
            cluster_type (Literal["cell", "pixel"]):
                the type of clustering being done


        Returns:
            MetaclusterColormap: The Dataclass containing the colormap-related information
        """

        cluster_id_to_name: pd.DataFrame = pd.read_csv(self.cluster_id_to_name_path)

        # The mapping file needs to contain the following columns:
        # 'cluster', 'metacluster', and 'mc_name'
        misc_utils.verify_same_elements(
            cluster_mapping_cols=cluster_id_to_name.columns.values,
            required_cols=[
                f"{self.cluster_type}_som_cluster",
                f"{self.cluster_type}_meta_cluster",
                f"{self.cluster_type}_meta_cluster_rename",
            ],
        )

        # subset on just metacluster and mc_name
        metacluster_id_to_name = cluster_id_to_name[
            [f"{self.cluster_type}_meta_cluster", f"{self.cluster_type}_meta_cluster_rename"]
        ].copy()

        unassigned_id: int = int(
            metacluster_id_to_name[f"{self.cluster_type}_meta_cluster"].max() + 1)

        # Extract unique pairs of (metacluster-ID,  name)
        # Set the unassigned cluster ID to be the max ID + 1
        # Set 0 as the Empty value
        metacluster_id_to_name: pd.DataFrame = pd.concat(
            [
                metacluster_id_to_name.drop_duplicates(),
                pd.DataFrame(
                    data={
                        f"{self.cluster_type}_meta_cluster": [unassigned_id, 0],
                        f"{self.cluster_type}_meta_cluster_rename": ["Unassigned", "Empty"]
                    }
                )
            ]
        )

        # sort by metacluster id ascending, this will help when making the colormap
        metacluster_id_to_name.sort_values(by=f'{self.cluster_type}_meta_cluster', inplace=True)

        # add the unassigned color to the metacluster_colors dict
        self.metacluster_colors.update({unassigned_id: self.unassigned_color})

        # add the no cluster color to the metacluster_colors dict
        self.metacluster_colors.update({0: self.no_cluster_color})

        # assert the metacluster index in colors matches with the ids in metacluster_id_to_name
        misc_utils.verify_same_elements(
            metacluster_colors_ids=list(
                self.metacluster_colors.keys()),
            metacluster_mapping_ids=metacluster_id_to_name
            [f'{self.cluster_type}_meta_cluster'].values)

        # use metacluster_colors to add the colors to metacluster_id_to_name
        metacluster_id_to_name["color"] = metacluster_id_to_name[
            f"{self.cluster_type}_meta_cluster"
        ].map(self.metacluster_colors)

        # Convert the list of tuples to a numpy array, each index is a color

        mc_colors: np.ndarray = np.array(metacluster_id_to_name['color'].to_list())

        metacluster_to_index = {}
        metacluster_id_to_name.reset_index(drop=True, inplace=True)
        for index, row in metacluster_id_to_name.reset_index(drop=True).iterrows():
            metacluster_to_index[row[f'{self.cluster_type}_meta_cluster']] = index

        # generate the colormap
        cmap = colors.ListedColormap(mc_colors)
        norm = colors.BoundaryNorm(
            np.linspace(0, len(mc_colors), len(mc_colors) + 1) - 0.5,
            len(mc_colors)
        )

        # Assign created values to dataclass attributes
        self.metacluster_id_to_name = metacluster_id_to_name
        self.mc_colors = mc_colors
        self.metacluster_to_index = metacluster_to_index
        self.cmap = cmap
        self.norm = norm

    def assign_metacluster_cmap(self, fov_img: np.ndarray) -> np.ndarray:
        """Assigns the metacluster colormap to the provided image.

        Args:
            fov_img (np.ndarray): The metacluster image to assign the colormap index to.

        Returns:
            np.ndarray: The image with the colormap index assigned.
        """
        # explicitly relabel each value in fov_img with its index in mc_colors
        # to ensure proper indexing into colormap
        relabeled_fov = np.copy(fov_img)
        for mc, mc_color_idx in self.metacluster_to_index.items():
            relabeled_fov[fov_img == mc] = mc_color_idx

        return relabeled_fov


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
    misc_utils.verify_in_list(fov_names=fovs, unique_fovs=img_xr.fovs.values)

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
                        cmap=cmap, norm=norm, interpolation='none')

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
                                    cluster_type='pixel', save_dir=None, fov_col='fovs',
                                    figsize=(10, 10)):
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
        cluster_type (str):
            the type of clustering being done
        save_dir (str):
            If provided, the image will be saved to this location.
        fov_col (str):
            column with the fovs names in `img_xr`.
        figsize (tuple):
            Size of the image that will be displayed.
    """

    # verify the type of clustering provided is valid
    misc_utils.verify_in_list(
        provided_cluster_type=[cluster_type],
        valid_cluster_types=['pixel', 'cell']
    )

    # verify the fovs are valid
    misc_utils.verify_in_list(fov_names=fovs, unique_fovs=img_xr.fovs.values)

    # verify cluster_id_to_name_path exists
    io_utils.validate_paths(cluster_id_to_name_path)

    # read the cluster to name mapping with the helper function
    mcc = MetaclusterColormap(cluster_type=cluster_type,
                              cluster_id_to_name_path=cluster_id_to_name_path,
                              metacluster_colors=metacluster_colors)
    for fov in fovs:
        # retrieve the image associated with the FOV
        fov_img = img_xr[img_xr[fov_col] == fov].values

        fov_img: np.ndarray = mcc.assign_metacluster_cmap(fov_img)

        # define the figure
        fig = plt.figure(figsize=figsize)

        # make the title
        plt.title(fov)

        # display the image
        overlay = plt.imshow(
            fov_img.squeeze(),
            cmap=mcc.cmap,
            norm=mcc.norm,
            origin='upper'
        )

        # remove the axes
        plt.axis('off')

        # remove the gridlines
        plt.grid(visible=False)

        # define the colorbar with annotations
        cax = fig.add_axes([0.9, 0.1, 0.01, 0.8])
        cbar = plt.colorbar(
            overlay,
            ticks=np.arange(len(mcc.mc_colors)),
            cax=cax,
            orientation='vertical'
        )
        cbar.ax.set_yticklabels(
            mcc.metacluster_id_to_name[f'{cluster_type}_meta_cluster_rename'].values)

        # explicitly turn off intermediate minor ticks
        for mt in cbar.ax.yaxis.get_minor_ticks():
            mt.set_visible(False)

        # save if specified
        if save_dir:
            misc_utils.save_figure(save_dir, f"{fov}.png")


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
            The path to the directory containing the segmentation data
        data_dir (str):
            The path to the directory containing the nuclear and whole cell image data
        img_overlay_chans (list):
            List of channels the user will overlay
        seg_overlay_comp (str):
            The segmented compartment the user will overlay
        alternate_segmentation (numpy.ndarray):
            2D numpy array of labeled cell objects
    Returns:
        numpy.ndarray:
            The image with the channel overlay
    """

    # load the specified fov data in
    plotting_tif = load_utils.load_imgs_from_dir(
        data_dir=data_dir,
        files=[fov + '.tiff'],
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
                                                             files=[fov + '_whole_cell.tiff'],
                                                             xr_dim_name='compartments',
                                                             xr_channel_names=['whole_cell'],
                                                             trim_suffix='_whole_cell',
                                                             match_substring='_whole_cell')
    segmentation_labels_nuc = load_utils.load_imgs_from_dir(data_dir=segmentation_dir,
                                                            files=[fov + '_nuclear.tiff'],
                                                            xr_dim_name='compartments',
                                                            xr_channel_names=['nuclear'],
                                                            trim_suffix='_nuclear',
                                                            match_substring='_nuclear')

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
                      seg_dir: Optional[Union[str, pathlib.Path]],
                      cluster_type='pixel',
                      mask_suffix: str = "_mask",
                      seg_suffix_name: Optional[str] = "_whole_cell.tiff",
                      img_sub_folder: str = ""):
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
        seg_dir (Union[str, pathlib.Path], optional):
            The location of the segmentation directory for the fovs. If None, then
            the segmentation file will not be copied over.
        cluster_type (str):
            the type of clustering being done
        mask_suffix (str, optional):
            The suffix used to find the mask tiffs. Defaults to "_mask".
        seg_suffix_name (str, optional):
            The suffix of the segmentation file and it's file extension. If None, then
            the segmentation file will not be copied over.
            Defaults to "_whole_cell.tiff".
        img_sub_folder (str, optional):
            The subfolder where the channels exist within the `img_data_path`.
            Defaults to "normalized".
    """

    # verify the type of clustering provided is valid
    misc_utils.verify_in_list(
        provided_cluster_type=[cluster_type],
        valid_cluster_types=['pixel', 'cell']
    )

    if not os.path.exists(mantis_project_path):
        os.makedirs(mantis_project_path)

    # create key from cluster number to cluster name
    if isinstance(mapping, (pathlib.Path, str)):
        map_df = pd.read_csv(mapping)
    elif isinstance(mapping, pd.DataFrame):
        map_df = mapping
    else:
        ValueError("Mapping must either be a path to an already saved mapping csv, \
                   or a DataFrame that is already loaded in.")

    # Save the segmentation tiff or not
    save_seg_tiff: bool = all(v is not None for v in [seg_dir, seg_suffix_name])

    map_df = map_df.loc[:, [f'{cluster_type}_meta_cluster', f'{cluster_type}_meta_cluster_rename']]
    # remove duplicates from df
    map_df = map_df.drop_duplicates()
    map_df = map_df.sort_values(by=[f'{cluster_type}_meta_cluster'])

    # rename for mantis names
    map_df = map_df.rename(
        {
            f'{cluster_type}_meta_cluster': 'region_id',
            f'{cluster_type}_meta_cluster_rename': 'region_name'
        },
        axis=1
    )

    # get names of fovs with masks
    mask_names_loaded = (io_utils.list_files(mask_output_dir, mask_suffix))
    mask_names_delimited = io_utils.extract_delimited_names(mask_names_loaded,
                                                            delimiter=mask_suffix)
    mask_names_sorted = natsort.natsorted(mask_names_delimited)

    # use `fovs`, a subset of the FOVs in `total_fov_names` which
    # is a list of FOVs in `img_data_path`
    fovs = natsort.natsorted(fovs)
    misc_utils.verify_in_list(fovs=fovs, img_data_fovs=mask_names_delimited)

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
            chans = io_utils.list_files(img_source_dir, substrs=EXTENSION_TYPES["IMAGE"])
            for chan in chans:
                shutil.copy(os.path.join(img_source_dir, chan), os.path.join(output_dir, chan))

        # copy mask into new folder
        mask_name: str = mn + mask_suffix + ".tiff"
        shutil.copy(os.path.join(mask_output_dir, mask_name),
                    os.path.join(output_dir, 'population{}.tiff'.format(mask_suffix)))

        # copy the segmentation files into the output directory
        # if `seg_dir` or `seg_name` is none, then skip copying
        if save_seg_tiff:
            seg_name: str = fov + seg_suffix_name
            shutil.copy(os.path.join(seg_dir, seg_name),
                        os.path.join(output_dir, 'cell_segmentation.tiff'))

        # copy mapping into directory
        map_df.to_csv(os.path.join(output_dir, 'population{}.csv'.format(mask_suffix)),
                      index=False)


def save_colored_masks(
        fovs: List[str],
        mask_dir: Union[str, pathlib.Path],
        save_dir: Union[str, pathlib.Path],
        cluster_id_to_name_path: Union[str, pathlib.Path],
        metacluster_colors: Dict,
        cluster_type: Literal["cell", "pixel"],
) -> None:
    """
    Converts the pixie TIFF masks into colored TIFF masks using the provided colormap and saves
    them in the `save_dir`. Mainly used for visualization purposes.

    Args:
        fovs (List[str]): A list of FOVs to save their associated color masks for.
        mask_dir (Union[str, pathlib.Path]): The directory where the pixie masks are stored.
        save_dir (Union[str, pathlib.Path]): The directory where the colored masks will be saved.
        cluster_id_to_name_path (Union[str, pathlib.Path]): A path to a CSV identifying the
            pixel/cell cluster to manually-defined name mapping this is output by the remapping
            visualization found in `metacluster_remap_gui`.
        metacluster_colors (Dict): Maps each metacluster id to a color.
        cluster_type (Literal["cell", "pixel"]): The type of clustering being done.
    """

    # Input validation
    misc_utils.verify_in_list(
        provided_cluster_type=[cluster_type],
        valid_cluster_types=["pixel", "cell"])

    # Create the save directory if it does not exist, convert mask and save dirs to Path objects
    if isinstance(mask_dir, str):
        mask_dir = pathlib.Path(mask_dir)
    if isinstance(save_dir, str):
        save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    io_utils.validate_paths([mask_dir, save_dir])

    mcc = MetaclusterColormap(cluster_type=cluster_type,
                              cluster_id_to_name_path=cluster_id_to_name_path,
                              metacluster_colors=metacluster_colors)

    with tqdm(total=len(fovs), desc="Saving colored masks", unit="FOVs") as pbar:
        for fov in fovs:
            mask: xr.DataArray = load_utils.load_imgs_from_dir(
                data_dir=mask_dir,
                files=[f"{fov}_{cluster_type}_mask.tiff"],
                trim_suffix=f"{cluster_type}_mask",
                match_substring=f"{cluster_type}_mask",
                xr_dim_name="pixel_mask",
                xr_channel_names=None,
            )

            # The values in the colored_mask are the indices of the colors in mcc.mc_colors
            # Make a new array with the actual colors, multiply by uint8 max to get 0-255 range

            colored_mask: np.ndarray = (mcc.mc_colors[mcc.assign_metacluster_cmap(
                mask.values.squeeze())] * 255.999).astype(np.uint8)

            image_utils.save_image(
                fname=save_dir / f"{fov}_{cluster_type}_mask_colored.tiff",
                data=colored_mask,)

            pbar.set_postfix(FOV=fov, refresh=False)
            pbar.update(1)
