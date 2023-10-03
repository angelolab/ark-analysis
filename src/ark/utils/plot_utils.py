import os
import pathlib
import shutil
from dataclasses import dataclass, field
from operator import contains
from typing import Dict, List, Literal, Optional, Tuple, Union
from matplotlib import gridspec
from matplotlib.axes import Axes

import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import colormaps, patches
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import skimage
import xarray as xr
from alpineer import image_utils, io_utils, load_utils, misc_utils
from alpineer.settings import EXTENSION_TYPES
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.exposure import rescale_intensity
from skimage import io


from skimage.util import img_as_ubyte
from tqdm.auto import tqdm
from ark import settings
from skimage.segmentation import find_boundaries
from ark.utils.data_utils import (
    ClusterMaskData,
    erode_mask,
    generate_cluster_mask,
    save_fov_mask,
    map_segmentation_labels,
)


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
    background_color: Tuple[float, ...] = field(init=False)
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

        # A pixel assigned to the background (black, #000000)
        self.background_color: Tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)

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
        self.metacluster_colors.update({0: self.background_color})

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


def create_cmap(cmap: Union[np.ndarray, list[str], str],
                n_clusters: int) -> tuple[colors.ListedColormap, colors.BoundaryNorm]:
    """
    Creates a discrete colormap and a boundary norm from the provided colors.

    Args:
        cmap (Union[np.ndarray, list[str], str]): The colormap, or set of colors to use.
        n_clusters (int): The numbe rof clusters for the colormap.

    Returns:
        tuple[colors.ListedColormap, colors.BoundaryNorm]:
            The generated colormap and boundary norm.
    """

    """Creates a colormap and a boundary norm from the provided colors.

    Colors can be of any format that matplotlib accepts.
    See here for color formats: https://matplotlib.org/stable/tutorials/colors/colors.html


    Args:
        colors_array (): The colors to use for the colormap.

    Returns:
        tuple[colors.ListedColormap, colors.BoundaryNorm]: The colormap and the boundary norm
    """

    if isinstance(cmap, np.ndarray):
        if cmap.ndim != 2:
            raise ValueError(
                f"colors_array must be a 2D array, got {cmap.ndim}D array")
        if cmap.shape[0] != n_clusters:
            raise ValueError(
                f"colors_array must have {n_clusters} colors, got {cmap.shape[0]} colors")
        color_map = colors.ListedColormap(colors=_cmap_add_background_unassigned(cmap))
    if isinstance(cmap, list):
        if len(cmap) != n_clusters:
            raise ValueError(
                f"colors_array must have {n_clusters} colors, got {len(cmap)} colors")
    if isinstance(cmap, str):
        try:
            # colorcet colormaps are also supported
            # cmocean colormaps are also supported
            color_map = colormaps[cmap]
        except KeyError:
            raise KeyError(f"Colormap {cmap} not found.")
        colors_rgba: np.ndarray = color_map(np.linspace(0, 1, n_clusters))
        color_map: colors.ListedColormap = colors.ListedColormap(
            colors=_cmap_add_background_unassigned(colors_rgba))

    bounds = [i-0.5 for i in np.linspace(0, color_map.N, color_map.N + 1)]

    norm = colors.BoundaryNorm(bounds, color_map.N)
    return color_map, norm


def _cmap_add_background_unassigned(cluster_colors: np.ndarray):
    # A pixel with no associated metacluster (gray, #5A5A5A)
    unassigned_color: np.ndarray = np.array([0.9, 0.9, 0.9, 1.0])

    # A pixel assigned to the background (black, #000000)
    background_color: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])

    return np.vstack([background_color, cluster_colors, unassigned_color])


def plot_cluster(
        image: np.ndarray,
        fov: str,
        cmap: colors.ListedColormap,
        norm: colors.BoundaryNorm,
        cbar_visible: bool = True,
        cbar_labels: list[str] = None,
        dpi: int = 300,
        figsize: tuple[int, int] = None) -> Figure:
    """
    Plots the cluster image with the provided colormap and norm.

    Args:
        image (np.ndarray):
            The cluster image to plot.
        fov (str):
            The name of the clustered FOV.
        cmap (colors.ListedColormap):
            A colormap to use for the cluster image.
        norm (colors.BoundaryNorm):
            A normalization to use for the cluster image.
        cbar_visible (bool, optional):
            Whether or not to display the colorbar. Defaults to True.
        cbar_labels (list[str], optional):
            Colorbar labels for the clusters. Devaults to None, where
            the labels will be automatically generated.
        dpi (int, optional):
            The resolution of the image to use for saving. Defaults to 300.
        figsize (tuple, optional):
            The size of the image to display. Defaults to (10, 10).

    Returns:
        Figure: Returns the cluster image as a matplotlib Figure.
    """
    # Default colorbar labels
    if cbar_labels is None:
        cbar_labels = [f"Cluster {x}" for x in range(1, len(cmap.colors))]

    fig: Figure = plt.figure(figsize=figsize, dpi=dpi)
    fig.set_layout_engine(layout="tight")
    gs = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)
    fig.suptitle(f"{fov}")

    # Image axis
    ax: Axes = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.grid(visible=False)

    ax.imshow(
        X=image,
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="equal",
        interpolation="none",
    )

    if cbar_visible:
        # # Manually set the colorbar
        divider = make_axes_locatable(fig.gca())
        cax = divider.append_axes(position="right", size="5%", pad="3%")

        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                            cax=cax, orientation="vertical", use_gridspec=True, pad=0.1,
                            shrink=0.9, drawedges=True)
        cbar.ax.set_yticks(
            ticks=np.arange(len(cbar_labels)),
            labels=cbar_labels
        )
        cbar.minorticks_off()

    return fig


def plot_neighborhood_cluster_result(img_xr: xr.DataArray,
                                     fovs: list[str],
                                     k: int,
                                     cmap_name: str = "tab20",
                                     cbar_visible: bool = True,
                                     save_dir: Union[str, pathlib.Path] = None,
                                     fov_col: str = "fovs",
                                     dpi: int = 300,
                                     figsize=(10, 10)
                                     ) -> None:
    """
    Plots the neighborhood clustering results for the provided FOVs.

    Args:
        img_xr (xr.DataArray):
            DataArray containing labeled cells.
        fovs (list[str]):
            A list of FOVs to plot.
        k (int):
            The number of neighborhoods / clusters.
        cmap_name (str, optional):
            The Colormap to use for clustering results. Defaults to "tab20".
        cbar_visible (bool, optional):
            Whether or not to display the colorbar. Defaults to True.
        save_dir (Union[str, pathlib.Path], optional):
            The image will be saved to this location if provided. Defaults to None.
        fov_col (str, optional):
            The column with the fov names in `img_xr`. Defaults to "fovs".
        dpi (int, optional):
            The resolution of the image to use for saving. Defaults to 300.
        figsize (tuple, optional):
            The size of the image to display. Defaults to (10, 10).
    """

    # verify the fovs are valid
    misc_utils.verify_in_list(fovs=fovs, unique_fovs=img_xr.fovs.values)

    # define the colormap
    my_colors = cm.get_cmap(cmap_name, k).colors

    cmap, norm = create_cmap(my_colors, n_clusters=k)

    cbar_labels = ["Empty"]
    cbar_labels.extend([f"Cluster {x}" for x in range(1, k+1)])

    for fov in img_xr.sel({fov_col: fovs}):

        fig: Figure = plot_cluster(
            image=fov.values.squeeze(),
            fov=fov.fovs.values,
            cmap=cmap,
            norm=norm,
            cbar_visible=cbar_visible,
            cbar_labels=cbar_labels,
            dpi=dpi,
            figsize=figsize
        )

        # save if specified
        if save_dir:
            fig.savefig(fname=os.path.join(save_dir, f"{fov.fovs.values}.png"), dpi=300)


def plot_pixel_cell_cluster(
        img_xr: xr.DataArray,
        fovs: list[str],
        cluster_id_to_name_path: Union[str, pathlib.Path],
        metacluster_colors: Dict,
        cluster_type: Union[Literal["pixel"], Literal["cell"]] = "pixel",
        cbar_visible: bool = True,
        save_dir=None,
        fov_col: str = "fovs",
        erode: bool = True,
        dpi=300,
        figsize=(10, 10)
):
    """Overlays the pixel and cell clusters on an image

    Args:
        img_xr (xr.DataArray):
            DataArray containing labeled pixel or cell clusters
        fovs (list[str]):
            A list of FOVs to plot.
        cluster_id_to_name_path (str):
            A path to a CSV identifying the pixel/cell cluster to manually-defined name mapping
            this is output by the remapping visualization found in `metacluster_remap_gui`
        metacluster_colors (dict):
            Dictionary which maps each metacluster id to a color
        cluster_type ("pixel" or "cell"):
            the type of clustering being done.
        cbar_visible (bool, optional):
            Whether or not to display the colorbar. Defaults to True.
        save_dir (str):
            If provided, the image will be saved to this location.
        fov_col (str):
            The column with the fov names in `img_xr`. Defaults to "fovs".
        erode (bool):
            Whether or not to erode the segmentation mask.
        dpi (int):
            The resolution of the image to use for saving. Defaults to 300.
        figsize (tuple):
            Size of the image that will be displayed.
    """

    # verify the type of clustering provided is valid
    misc_utils.verify_in_list(
        provided_cluster_type=[cluster_type],
        valid_cluster_types=['pixel', 'cell']
    )

    # verify the fovs are valid
    misc_utils.verify_in_list(fovs=fovs, unique_fovs=img_xr.fovs.values)

    # verify cluster_id_to_name_path exists
    io_utils.validate_paths(cluster_id_to_name_path)

    # read the cluster to name mapping with the helper function
    mcc = MetaclusterColormap(cluster_type=cluster_type,
                              cluster_id_to_name_path=cluster_id_to_name_path,
                              metacluster_colors=metacluster_colors)

    for fov in img_xr.sel({fov_col: fovs}):
        fov_name = fov.fovs.values
        if erode:
            fov = erode_mask(seg_mask=fov, connectivity=2, mode="thick", background=0)

        fov_img = mcc.assign_metacluster_cmap(fov_img=fov)

        fig: Figure = plot_cluster(
            image=fov_img,
            fov=fov_name,
            cmap=mcc.cmap,
            norm=mcc.norm,
            cbar_visible=cbar_visible,
            cbar_labels=mcc.metacluster_id_to_name[f'{cluster_type}_meta_cluster_rename'].values,
            dpi=dpi,
            figsize=figsize,
        )

        # save if specified
        if save_dir:
            fig.savefig(fname=os.path.join(save_dir, f"{fov_name}.png"), dpi=300)


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
                      img_sub_folder: str = None,
                      new_mask_suffix: str = None):
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
            Defaults to None.
        new_mask_suffix (str, optional):
            The new suffix added to the copied mask tiffs.
    """

    # verify the type of clustering provided is valid
    misc_utils.verify_in_list(
        provided_cluster_type=[cluster_type],
        valid_cluster_types=['pixel', 'cell']
    )

    if not os.path.exists(mantis_project_path):
        os.makedirs(mantis_project_path)

    # account for non-sub folder channel file structures
    img_sub_folder = "" if not img_sub_folder else img_sub_folder

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

    # if no new suffix specified, copy over with original mask name
    if not new_mask_suffix:
        new_mask_suffix = mask_suffix

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

    # use `fovs`, a subset of the FOVs in `total_fovs` which
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
                    os.path.join(output_dir, 'population{}.tiff'.format(new_mask_suffix)))

        # copy the segmentation files into the output directory
        # if `seg_dir` or `seg_name` is none, then skip copying
        if save_seg_tiff:
            if not os.path.exists(os.path.join(output_dir, 'cell_segmentation.tiff')):
                seg_name: str = fov + seg_suffix_name
                shutil.copy(os.path.join(seg_dir, seg_name),
                            os.path.join(output_dir, 'cell_segmentation.tiff'))

        # copy mapping into directory
        map_df.to_csv(os.path.join(output_dir, 'population{}.csv'.format(new_mask_suffix)),
                      index=False)


def save_colored_mask(
    fov: str,
    save_dir: str,
    suffix: str,
    data: np.ndarray,
    cmap: colors.ListedColormap,
    norm: colors.BoundaryNorm,
) -> None:
    """Saves the colored mask to the provided save directory.

    Args:
        fov (str):
            The name of the FOV.
        save_dir (str):
            The directory where the colored mask will be saved.
        suffix (str):
            The suffix to append to the FOV name.
        data (np.ndarray):
            The mask to save.
        cmap (colors.ListedColormap):
            The colormap to use for the mask.
        norm (colors.BoundaryNorm):
            The normalization to use for the mask.
    """

    # Create the save directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the colored mask
    colored_mask = img_as_ubyte(cmap(norm(data)))

    # Save the image
    image_utils.save_image(
        fname=os.path.join(save_dir, f"{fov}{suffix}"),
        data=colored_mask,
    )


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
            pbar.set_postfix(FOV=fov, refresh=False)

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

            pbar.update(1)


def cohort_cluster_plot(
    fovs: List[str],
    seg_dir: Union[pathlib.Path, str],
    save_dir: Union[pathlib.Path, str],
    cell_data: pd.DataFrame,
    fov_col: str = settings.FOV_ID,
    label_col: str = settings.CELL_LABEL,
    cluster_col: str = settings.CELL_TYPE,
    seg_suffix: str = "_whole_cell.tiff",
    cmap: Union[str, pd.DataFrame] = "viridis",
    style: str = "seaborn-v0_8-paper",
    erode: bool = False,
    display_fig: bool = False,
    fig_file_type: str = "png",
    figsize: tuple = (10, 10),
    dpi: int = 300,
) -> None:
    """
    Saves the cluster masks for each FOV in the cohort as the following:
    - Cluster mask numbered 1-N, where N is the number of clusters (tiff)
    - Cluster mask colored by cluster with or without a colorbar (png)
    - Cluster mask colored by cluster (tiff).

    Args:
        fovs (List[str]): A list of FOVs to generate cluster masks for.
        seg_dir (Union[pathlib.Path, str]): The directory containing the segmentation masks.
        save_dir (Union[pathlib.Path, str]): The directory to save the cluster masks to.
        cell_data (pd.DataFrame): The cell data table containing the cluster labels.
        fov_col (str, optional): The column containing the FOV name. Defaults to settings.FOV_ID.
        label_col (str, optional): The column containing the segmentaiton label.
            Defaults to settings.CELL_LABEL.
        cluster_col (str, optional): The column containing the cluster a segmentation label
            belongs to. Defaults to settings.CELL_TYPE.
        seg_suffix (str, optional): The kind of segmentation file to read.
            Defaults to "_whole_cell.tiff".
        cmap (str, pd.DataFrame, optional): The colormap to generate clusters from,
            or a DataFrame, where the user can specify their own colors per cluster.
            The color column must be labeled "color". Defaults to "viridis".
        style (str, optional): Set the matplotlib style image style. Defaults to 
            "seaborn-v0_8-paper".
            View the available styles here: 
            https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
            Or run matplotlib.pyplot.style.available in a notebook to view all the styles.
        erode (bool, optional): Option to "thicken" the cell boundary via the segmentation label
            for visualization purposes. Defaults to False.
        display_fig (bool, optional): Option to display the cluster mask plots as they are
            generated. Defaults to False. Displaying each figure can use a lot of memory,
            so it's best to try to visualize just a few FOVs, before generating the cluster masks
            for the entire cohort.
        fig_file_type (str, optional): The file type to save figures as. Defaults to 'png'.
        figsize (tuple, optional):
            The size of the figure to display. Defaults to (10, 10).
        dpi (int, optional):
            The resolution of the image to use for saving. Defaults to 300.
    """

    plt.style.use(style)

    if isinstance(seg_dir, str):
        seg_dir = pathlib.Path(seg_dir)

    try:
        io_utils.validate_paths(seg_dir)
    except ValueError:
        raise ValueError(f"Could not find the segmentation directory at {seg_dir.as_posix()}")

    if isinstance(save_dir, str):
        save_dir = pathlib.Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(fovs, str):
        fovs = [fovs]

    # Create the subdirectories for the 3 cluster mask files
    for sub_dir in ["cluster_masks", "cluster_masks_colored", "cluster_plots"]:
        (save_dir / sub_dir).mkdir(parents=True, exist_ok=True)

    cmd = ClusterMaskData(
        data=cell_data,
        fov_col=fov_col,
        label_col=label_col,
        cluster_col=cluster_col,
    )
    if isinstance(cmap, pd.DataFrame):
        unique_clusters: pd.DataFrame = cmd.mapping[[cmd.cluster_column,
                                                     cmd.cluster_id_column]].drop_duplicates()
        cmap_colors: pd.DataFrame = cmap.merge(
            right=unique_clusters,
            on=cmd.cluster_column
        ).sort_values(by="cluster_id")["color"].values
        colors_like: list[bool] = [colors.is_color_like(c) for c in cmap_colors]

        if not all(colors_like):
            bad_color_values: np.ndarray = cmap_colors[~np.array(colors_like)]
            raise ValueError(
                ("Not all colors in the provided cmap are valid colors."
                 f"The following colors are invalid: {bad_color_values}"))

        np_colors = colors.to_rgba_array(cmap_colors)

        color_map, norm = create_cmap(np_colors, n_clusters=cmd.n_clusters)

    if isinstance(cmap, str):
        color_map, norm = create_cmap(cmap, n_clusters=cmd.n_clusters)

    # create the pixel cluster masks across each fov
    with tqdm(total=len(fovs), desc="Cluster Mask Generation", unit="FOVs") as pbar:
        for fov in fovs:
            pbar.set_postfix(FOV=fov)

            # generate the cell mask for the FOV
            cluster_mask: np.ndarray = generate_cluster_mask(
                fov=fov,
                seg_dir=seg_dir,
                cmd=cmd,
                seg_suffix=seg_suffix,
                erode=erode,
            )

            # save the cluster mask generated
            save_fov_mask(
                fov,
                data_dir=save_dir / "cluster_masks",
                mask_data=cluster_mask,
                sub_dir=None,
            )

            save_colored_mask(
                fov=fov,
                save_dir=save_dir / "cluster_masks_colored",
                suffix=".tiff",
                data=cluster_mask,
                cmap=color_map,
                norm=norm,
            )

            cluster_labels = ["Background"] + cmd.cluster_names + ["Unassigned"]

            fig = plot_cluster(
                image=cluster_mask,
                fov=fov,
                cmap=color_map,
                norm=norm,
                cbar_visible=True,
                cbar_labels=cluster_labels,
                figsize=figsize,
                dpi=dpi,
            )

            fig.savefig(
                fname=os.path.join(save_dir, "cluster_plots", f"{fov}.{fig_file_type}"),
            )

            if display_fig:
                fig.show(warn=False)
            else:
                plt.close(fig)

            pbar.update(1)


def plot_continuous_variable(
    image: np.ndarray,
    name: str,
    stat_name: str,
    cmap: Union[colors.Colormap, str],
    norm: colors.Normalize = None,
    cbar_visible: bool = True,
    dpi: int = 300,
    figsize: tuple[int, int] = (10, 10),
) -> Figure:
    """

    Plots an image measuring some type of continuous variable with a user provided colormap.

    Args:
        image (np.ndarray):
            An array representing an image to plot.
        name (str):
            The name of the image.
        stat_name (str):
            The name of the statistic to plot, this will be the colormap's label.
        cmap (colors.Colormap, str, optional): A colormap to plot the array with.
            Defaults to "viridis".
        cbar_visible (bool, optional): A flag for setting the colorbar on or not.
            Defaults to True.
        norm (colors.Normalize, optional): A normalization to apply to the colormap.
        dpi (int, optional):
            The resolution of the image. Defaults to 300.
        figsize (tuple[int, int], optional):
            The size of the image. Defaults to (10, 10).

    Returns:
        Figure : The Figure object of the image.
    """
    fig: Figure = plt.figure(figsize=figsize, dpi=dpi)
    fig.set_layout_engine(layout="tight")
    gs = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)
    fig.suptitle(f"{name}")

    # Image axis
    ax: Axes = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.grid(visible=False)

    im = ax.imshow(
        X=image,
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="equal",
        interpolation="none",
    )

    if cbar_visible:
        # Manually set the colorbar
        divider = make_axes_locatable(fig.gca())
        cax = divider.append_axes(position="right", size="5%", pad="3%")

        fig.colorbar(mappable=im, cax=cax, orientation="vertical",
                     use_gridspec=True, pad=0.1, shrink=0.9, drawedges=False, label=stat_name)

    return fig


def color_segmentation_by_stat(
    fovs: List[str],
    data_table: pd.DataFrame,
    seg_dir: Union[pathlib.Path, str],
    save_dir: Union[pathlib.Path, str],
    fov_col: str = settings.FOV_ID,
    label_col: str = settings.CELL_LABEL,
    stat_name: str = settings.CELL_TYPE,
    cmap: str = "viridis",
    reverse: bool = False,
    seg_suffix: str = "_whole_cell.tiff",
    cbar_visible: bool = True,
    style: str = "seaborn-v0_8-paper",
    erode: bool = False,
    display_fig: bool = False,
    fig_file_type: str = "png",
    figsize: tuple = (10, 10),
    dpi: int = 300,
):
    """
    Colors segmentation masks by a given continuous statistic.

    Args:
        fovs: (List[str]):
            A list of FOVs to plot.
        data_table (pd.DataFrame):
            A DataFrame containing FOV and segmentation label identifiers
            as well as a collection of statistics for each label in a segmentation
            mask such as:

                - `fov_id` (identifier)
                - `label` (identifier)
                - `area` (statistic)
                - `fiber` (statistic)
                - etc...

        seg_dir (Union[pathlib.Path, str]):
            Path to the directory containing segmentation masks.
        save_dir (Union[pathlib.Path, str]):
            Path to the directory where the colored segmentation masks will be saved.
        fov_col: (str, optional):
            The name of the column in `data_table` containing the FOV identifiers.
            Defaults to "fov".
        label_col (str, optional):
            The name of the column in `data_table` containing the segmentation label identifiers.
            Defaults to "label".
        stat_name (str):
            The name of the statistic to color the segmentation masks by. This should be a column
            in `data_table`.
        seg_suffix (str, optional):
            The suffix of the segmentation file and it's file extension. Defaults to
            "_whole_cell.tiff".
        cmap (str, optional): The colormap for plotting. Defaults to "viridis".
        reverse (bool, optional):
            A flag to reverse the colormap provided. Defaults to False.
        cbar_visible (bool, optional):
            A flag to display the colorbar. Defaults to True.
        erode (bool, optional): Option to "thicken" the cell boundary via the segmentation label
            for visualization purposes. Defaults to False.
        style (str, optional): Set the matplotlib style image style. Defaults to 
            "seaborn-v0_8-paper".
            View the available styles here: 
            https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
            Or run matplotlib.pyplot.style.available in a notebook to view all the styles.
        display_fig: (bool, optional):
            Option to display the cluster mask plots as they are generated. Defaults to False.
        fig_file_type (str, optional): The file type to save figures as. Defaults to 'png'.
        figsize (tuple, optional):
            The size of the figure to display. Defaults to (10, 10).
        dpi (int, optional):
            The resolution of the image to use for saving. Defaults to 300.
    """
    plt.style.use(style)

    if not isinstance(seg_dir, pathlib.Path):
        seg_dir = pathlib.Path(seg_dir)

    if not isinstance(save_dir, pathlib.Path):
        save_dir = pathlib.Path(save_dir)

    io_utils.validate_paths([seg_dir])

    try:
        io_utils.validate_paths([save_dir])
    except FileNotFoundError:
        save_dir.mkdir(parents=True, exist_ok=True)

    misc_utils.verify_in_list(
        statistic_name=[fov_col, label_col, stat_name],
        data_table_columns=data_table.columns,
    )

    if not (save_dir / "continuous_plots").exists():
        (save_dir / "continuous_plots").mkdir(parents=True, exist_ok=True)
    if not (save_dir / "colored").exists():
        (save_dir / "colored").mkdir(parents=True, exist_ok=True)

    # filter the data table to only include the FOVs we want to plot
    data_table = data_table[data_table[fov_col].isin(fovs)]

    data_table_subset_groups: DataFrameGroupBy = (
        data_table[[fov_col, label_col, stat_name]]
        .sort_values(by=[fov_col, label_col], key=natsort.natsort_keygen())
        .groupby(by=fov_col)
    )

    # Colormap normalization across the cohort + reverse if necessary
    vmin: np.float64 = data_table[stat_name].min()
    vmax: np.float64 = data_table[stat_name].max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    if reverse:
        # Adding the suffix "_r" will reverse the colormap
        cmap = f"{cmap}_r"

    # Prepend black to the colormap
    color_map = set_minimum_color_for_colormap(
        cmap=colormaps[cmap], default=(0, 0, 0, 1)
    )

    with tqdm(
        total=len(data_table_subset_groups),
        desc=f"Generating {stat_name} Plots",
        unit="FOVs",
    ) as pbar:
        for fov, fov_group in data_table_subset_groups:
            pbar.set_postfix(FOV=fov)

            label_map: np.ndarray = io.imread(seg_dir / f"{fov}{seg_suffix}")

            if erode:
                label_map = erode_mask(
                    label_map, connectivity=2, mode="thick", background=0
                )

            mapped_seg_image: np.ndarray = map_segmentation_labels(
                labels=fov_group[label_col],
                values=fov_group[stat_name],
                label_map=label_map,
            )

            fig = plot_continuous_variable(
                image=mapped_seg_image,
                name=fov,
                stat_name=stat_name,
                norm=norm,
                cmap=color_map,
                cbar_visible=cbar_visible,
                figsize=figsize,
                dpi=dpi,
            )
            fig.savefig(fname=os.path.join(save_dir, "continuous_plots", f"{fov}.{fig_file_type}"))

            save_colored_mask(
                fov=fov,
                save_dir=save_dir / "colored",
                suffix=".tiff",
                data=mapped_seg_image,
                cmap=color_map,
                norm=norm,
            )
            if display_fig:
                fig.show(warn=False)
            else:
                plt.close(fig)

            pbar.update(1)
