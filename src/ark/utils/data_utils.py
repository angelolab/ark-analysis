import numba as nb
import itertools
import os
import pathlib
import re
from typing import List, Literal, Union, Sequence

from numpy.typing import ArrayLike, DTypeLike
from numpy import ma
import feather
import natsort as ns
import numpy as np
import pandas as pd
import skimage.io as io
from alpineer import data_utils, image_utils, io_utils, load_utils, misc_utils
from alpineer.settings import EXTENSION_TYPES
from tqdm.notebook import tqdm_notebook as tqdm
import xarray as xr
from ark import settings
from skimage.segmentation import find_boundaries
from pandas.core.groupby.generic import DataFrameGroupBy
from anndata import AnnData, read_zarr
from anndata.experimental import AnnCollection
from anndata.experimental.multi_files._anncollection import ConvertType
from torchdata.datapipes.iter import IterDataPipe
from typing import Iterator, Optional
try:
    from typing import TypedDict, Unpack
except ImportError:
    from typing_extensions import TypedDict, Unpack


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


def erode_mask(seg_mask: np.ndarray, **kwargs) -> np.ndarray:
    """
    Erodes the edges labels of a segmentation mask.
    Other keyword arguments get passed to `skimage.segmentation.find_boundaries`.

    Args:
        seg_mask (np.ndarray): The segmentation mask to erode.

    Returns:
        np.ndarray: The eroded segmentation mask
    """
    edges = find_boundaries(
        label_img=seg_mask, **kwargs)
    seg_mask = np.where(edges == 0, seg_mask, 0)
    return seg_mask


class ClusterMaskData:
    """
    A class containing the cell labels, cluster labels, and segmentation labels for the
    whole cohort. Also contains the mapping from the segmentation label to the cluster
    label for each FOV.
    """

    fov_column: str
    label_column: str
    cluster_column: str
    unique_fovs: List[str]
    cluster_id_column: str
    unassigned_id: int
    n_clusters: int
    mapping: pd.DataFrame

    def __init__(
        self, data: pd.DataFrame, fov_col: str, label_col: str, cluster_col: str
    ) -> None:
        """
        A class containing the cell data, cell label column, cluster column and the mapping from a
        cell label to a cluster.

        Args:
            data (pd.DataFrame):
                A cell table with the cell label column and the cluster column.
            fov_col (str):
                The name of the column in the cell table that contains the FOV ID.
            label_col (str):
                The name of the column in the cell table that contains the cell label.
            cluster_col (str):
                The name of the column in the cell table that contains the cluster label.
        """
        self.fov_column: str = fov_col
        self.label_column: str = label_col
        self.cluster_column: str = cluster_col
        self.cluster_id_column: str = "cluster_id"

        # Extract only the necessary columns: fov ID, segmentation label, cluster label
        mapping_data: pd.DataFrame = data[
            [self.fov_column, self.label_column, self.cluster_column]
        ].copy()

        # Add a cluster_id_column to the column in case the cluster_column is
        # non-numeric (i.e. string), index in ascending order of cell_meta_cluster
        cluster_name_id = pd.DataFrame(
            {self.cluster_column: mapping_data[self.cluster_column].unique()})
        cluster_name_id.sort_values(by=f'{self.cluster_column}', inplace=True)
        cluster_name_id.reset_index(drop=True, inplace=True)

        cluster_name_id[self.cluster_id_column] = (cluster_name_id.index + 1).astype(np.int32)

        self.cluster_name_id = cluster_name_id

        # merge the cluster_id_column to the mapping_data dataframe
        mapping_data = mapping_data.merge(right=self.cluster_name_id, on=self.cluster_column)

        mapping_data = mapping_data.astype(
            {
                self.fov_column: str,
                self.label_column: np.int32,
                self.cluster_id_column: np.int32,
            }
        )
        self.unique_fovs: List[str] = ns.natsorted(
            mapping_data[self.fov_column].unique().tolist()
        )

        self.unassigned_id: np.int32 = np.int32(
            mapping_data[self.cluster_id_column].max() + 1
        )
        self.n_clusters: int = mapping_data[self.cluster_id_column].max()

        # For each FOV map the segmentation label 0 (background) to the cluster label 0
        cluster0_mapping: pd.DataFrame = pd.DataFrame(
            data={
                self.fov_column: self.unique_fovs,
                self.label_column: np.repeat(0, repeats=len(self.unique_fovs)),
                self.cluster_column: np.repeat(0, repeats=len(self.unique_fovs)),
                self.cluster_id_column: np.repeat(0, repeats=len(self.unique_fovs)),
            }
        )

        mapping_data = pd.concat(objs=[mapping_data, cluster0_mapping]).astype(
            {
                self.fov_column: str,
                self.label_column: np.int32,
                self.cluster_id_column: np.int32,
            }
        )

        # Sort by FOV first, then by segmentation label
        self.mapping = mapping_data.sort_values(by=[self.fov_column, self.label_column])

    def fov_mapping(self, fov: str) -> pd.DataFrame:
        """Returns the mapping for a specific FOV.
        Args:
            fov (str):
                The FOV to get the mapping for.
        Returns:
            pd.DataFrame:
                The mapping for the FOV.
        """
        misc_utils.verify_in_list(requested_fov=[fov], all_fovs=self.unique_fovs)
        fov_data: pd.DataFrame = self.mapping[self.mapping[self.fov_column] == fov]
        return fov_data.reset_index(drop=True)

    @property
    def cluster_names(self) -> List[str]:
        """Returns the cluster names.
        Returns:
            List[str]:
                The cluster names.
        """
        return self.cluster_name_id[self.cluster_column].tolist()


def label_cells_by_cluster(
        fov: str,
        cmd: ClusterMaskData,
        label_map: Union[np.ndarray, xr.DataArray],
) -> np.ndarray:
    """Translates cell-ID labeled images according to the clustering assignment
    found in cell_cluster_mask_data.


    Args:
        fov (str):
            The FOV to relabel
        cmd (ClusterMaskData):
            A dataclass containing the cell data, cell label column, cluster column and the
            mapping from the segmentation label to the cluster label for a given FOV.
        label_map (xarray.DataArray):
            label map for a single FOV.

    Returns:
        numpy.ndarray:
            The image with new designated label assignments
    """

    # verify that fov found in all_data
    misc_utils.verify_in_list(
        fov_name=[fov],
        all_data_fovs=cmd.unique_fovs
    )

    # condense extraneous axes if label_map is a DataArray
    if isinstance(label_map, xr.DataArray):
        labeled_image = label_map.squeeze().values.astype(np.int32)
    else:
        labeled_image: np.ndarray = label_map.squeeze().astype(np.int32)

    fov_clusters: pd.DataFrame = cmd.fov_mapping(fov=fov)

    mapping: nb.typed.typeddict = nb.typed.Dict.empty(
        key_type=nb.types.int32,
        value_type=nb.types.int32,
    )

    for label, cluster in fov_clusters[[cmd.label_column, cmd.cluster_id_column]].itertuples(
            index=False):
        mapping[np.int32(label)] = np.int32(cluster)

    relabeled_image: np.ndarray = relabel_segmentation(
        mapping=mapping,
        unassigned_id=cmd.unassigned_id,
        labeled_image=labeled_image,
        _dtype=np.int32)

    return relabeled_image.astype(np.int16)


def map_segmentation_labels(
    labels: Union[pd.Series, np.ndarray],
    values: Union[pd.Series, np.ndarray],
    label_map: ArrayLike,
    unassigned_id: float = 0,
) -> np.ndarray:
    """
    Maps an image consisting of segmentation labels to an image consisting of a particular type of
    statistic, metric, or value of interest.

    Args:
        labels (Union[pd.Series, np.ndarray]): The segmentation labels.
        values (Union[pd.Series, np.ndarray]): The values to map to the segmentation labels.
        label_map (ArrayLike): The segmentation labels as an image to map to.
        unassigned_id (int | float, optional): A default value to assign there is exists no 1-to-1
        mapping from a label in the label_map to a label in the `labels` argument. Defaults to 0.

    Returns:
        np.ndarray: Returns the mapped image.
    """
    # condense extraneous axes if label_map is a DataArray
    if isinstance(label_map, xr.DataArray):
        labeled_image = label_map.squeeze().values.astype(np.int32)
    else:
        labeled_image: np.ndarray = label_map.squeeze().astype(np.int32)

    if isinstance(labels, pd.Series):
        labels = labels.to_numpy(dtype=np.int32)
    if isinstance(values, pd.Series):
        # handle NaNs, replace with 0
        values = ma.fix_invalid(values.to_numpy(dtype=np.float64), fill_value=0).data

    mapping: nb.typed.typeddict = nb.typed.Dict.empty(
        key_type=nb.types.int32, value_type=nb.types.float64
    )

    for label, value in zip(labels, values):
        mapping[label] = value

    relabeled_image: np.ndarray = relabel_segmentation(
        mapping=mapping,
        unassigned_id=unassigned_id,
        labeled_image=labeled_image,
        _dtype=np.float64,
    )

    return relabeled_image


@nb.njit(parallel=True)
def relabel_segmentation(
    mapping: nb.typed.typeddict,
    unassigned_id: np.int32,
    labeled_image: np.ndarray,
    _dtype: DTypeLike = np.float64,
) -> np.ndarray:
    """
    Relabels a labled segmentation image according to the provided values.

    Args:
        mapping (nb.typed.typeddict):
            A Numba typed dictionary mapping segmentation labels to cluster labels.
        unassigned_id (np.int32):
            The label given to a pixel with no associated cluster.
        labeled_image (np.ndarray):
            The labeled segmentation image.
        _dtype (DTypeLike, optional):
            The data type of the relabeled image. Defaults to `np.float64`.

    Returns:
        np.ndarray: The relabeled segmentation image.
    """
    relabeled_image: np.ndarray = np.empty(shape=labeled_image.shape, dtype=_dtype)
    for i in nb.prange(labeled_image.shape[0]):
        for j in nb.prange(labeled_image.shape[1]):
            relabeled_image[i, j] = mapping.get(labeled_image[i, j], unassigned_id)
    return relabeled_image


def generate_cluster_mask(
        fov: str,
        seg_dir: Union[str, pathlib.Path],
        cmd: ClusterMaskData,
        seg_suffix: str = "_whole_cell.tiff",
        erode: bool = True,
        **kwargs) -> np.ndarray:
    """For a fov, create a mask labeling each cell with their SOM or meta cluster label

    Args:
        fov (str):
            The fov to relabel
        seg_dir (str):
            The path to the segmentation data
        cmd (ClusterMaskData):
            A dataclass containing the cell data, cell label column, cluster column and the
            mapping from the segmentation label to the cluster label for a given FOV.
        seg_suffix (str):
            The suffix that the segmentation images use. Defaults to `'_whole_cell.tiff'`.
        erode (bool):
            Whether to erode the edges of the segmentation mask. Defaults to `True`.

    Returns:
        numpy.ndarray:
            The image where values represent cell cluster labels.
    """

    # path checking
    io_utils.validate_paths([seg_dir])

    # define the file for whole cell
    whole_cell_files = [fov + seg_suffix]

    # load the segmentation labels in for the FOV
    label_map = load_utils.load_imgs_from_dir(
        data_dir=seg_dir, files=whole_cell_files, xr_dim_name='compartments',
        xr_channel_names=['whole_cell'], trim_suffix=seg_suffix.split('.')[0]
    ).loc[fov, ...]

    if erode:
        label_map = erode_mask(label_map, connectivity=2, mode="thick", background=0)

    # use label_cells_by_cluster to create cell masks
    img_data: np.ndarray = label_cells_by_cluster(
        fov=fov,
        cmd=cmd,
        label_map=label_map
    )

    return img_data


def generate_and_save_cell_cluster_masks(
    fovs: List[str],
    save_dir: Union[pathlib.Path, str],
    seg_dir: Union[pathlib.Path, str],
    cell_data: pd.DataFrame,
    cluster_id_to_name_path: Union[pathlib.Path, str],
    fov_col: str = settings.FOV_ID,
    label_col: str = settings.CELL_LABEL,
    cell_cluster_col: str = settings.CELL_TYPE,
    seg_suffix: str = "_whole_cell.tiff",
    sub_dir: str = None,
    name_suffix: str = "",
):
    """Generates cell cluster masks and saves them for downstream analysis.

    Args:
        fovs (List[str]):
            A list of fovs to generate and save pixel masks for.
        save_dir (Union[pathlib.Path, str]):
            The directory to save the generated cell cluster masks.
        seg_dir (Union[pathlib.Path, str]):
            The path to the segmentation data.
        cell_data (pd.DataFrame):
            The cell data with both cell SOM and meta cluster assignments.
        cluster_id_to_name_path (Union[str, pathlib.Path]): A path to a CSV identifying the
            cell cluster to manually-defined name mapping this is output by the remapping
            visualization found in `metacluster_remap_gui`.
        fov_col (str, optional):
            The column name containing the FOV IDs . Defaults to `settings.FOV_ID` (`"fov"`).
        label_col (str, optional):
            The column name containing the cell label. Defaults to
            `settings.CELL_LABEL` (`"label"`).
        cell_cluster_col (str, optional):
            Whether to assign SOM or meta clusters. Needs to be `"cell_som_cluster"` or
            `"cell_meta_cluster"`. Defaults to `settings.CELL_TYPE` (`"cell_meta_cluster"`).
        seg_suffix (str, optional):
            The suffix that the segmentation images use. Defaults to `"_whole_cell.tiff"`.
        sub_dir (str, optional):
            The subdirectory to save the images in. If specified images are saved to
            `"data_dir/sub_dir"`. If `sub_dir = None` the images are saved to `"data_dir"`.
            Defaults to `None`.
        name_suffix (str, optional):
            Specify what to append at the end of every cell mask. Defaults to `""`.
    """

    cmd = ClusterMaskData(
        data=cell_data,
        fov_col=fov_col,
        label_col=label_col,
        cluster_col=cell_cluster_col,
    )

    # read in gui cluster mapping file and new cluster mapping created by ClusterMaskData
    gui_map = pd.read_csv(cluster_id_to_name_path)
    cluster_map = cmd.mapping.filter([cmd.cluster_column, cmd.cluster_id_column])
    cluster_map = cluster_map.drop_duplicates()

    # drop the cluster_id column from updated_cluster_map if it already exists, otherwise do nothing
    gui_map = gui_map.drop(columns="cluster_id", errors="ignore")

    # add a cluster_id column corresponding to the new mask integers
    updated_cluster_map = gui_map.merge(cluster_map, on=[cmd.cluster_column], how="left")
    updated_cluster_map.to_csv(cluster_id_to_name_path, index=False)

    # create the pixel cluster masks across each fov
    with tqdm(total=len(fovs), desc="Cell Cluster Mask Generation", unit="FOVs") as pbar:
        for fov in fovs:
            pbar.set_postfix(FOV=fov)

            # generate the cell mask for the FOV
            cell_mask: np.ndarray = generate_cluster_mask(
                fov=fov, seg_dir=seg_dir, cmd=cmd, seg_suffix=seg_suffix
            )

            # save the cell mask generated
            save_fov_mask(
                fov,
                data_dir=save_dir,
                mask_data=cell_mask,
                sub_dir=sub_dir,
                name_suffix=name_suffix,
            )

            pbar.update(1)


def generate_pixel_cluster_mask(fov, base_dir, tiff_dir, chan_file_path,
                                pixel_data_dir, cluster_mapping,
                                pixel_cluster_col='pixel_meta_cluster'):
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
        cluster_mapping (pd.DataFrame)
            Dataframe detailing which meta_cluster IDs map to which cluster_id
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
    coordinates = x_coords * img_data.shape[1] + y_coords

    # get the corresponding cluster labels for each pixel
    cluster_labels = list(fov_data[pixel_cluster_col])

    # relabel meta_cluster numbers with cluster_id
    cluster_mapping = cluster_mapping.drop_duplicates()[[pixel_cluster_col, 'cluster_id']]
    id_mapping = dict(zip(cluster_mapping[pixel_cluster_col], cluster_mapping['cluster_id']))
    cluster_labels = [id_mapping[label] for label in cluster_labels]

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
                                          cluster_id_to_name_path: Union[pathlib.Path, str],
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
        cluster_id_to_name_path (Union[str, pathlib.Path]): A path to a CSV identifying the
            pixel cluster to manually-defined name mapping this is output by the remapping
            visualization found in `metacluster_remap_gui`.
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
    # read in gui cluster mapping file and save cluster_id created in generate_pixel_cluster_mask
    gui_map = pd.read_csv(cluster_id_to_name_path)
    cluster_map = gui_map.copy()[[pixel_cluster_col]]

    cluster_map = cluster_map.drop_duplicates().sort_values(by=[pixel_cluster_col])
    cluster_map["cluster_id"] = list(range(1, len(cluster_map) + 1))

    # drop the cluster_id column from gui_map if it already exists, otherwise do nothing
    gui_map = gui_map.drop(columns="cluster_id", errors="ignore")

    # add a cluster_id column corresponding to the new mask integers
    updated_cluster_map = gui_map.merge(cluster_map, on=[pixel_cluster_col], how="left")
    updated_cluster_map.to_csv(cluster_id_to_name_path, index=False)

    # create the pixel cluster masks across each fov
    with tqdm(total=len(fovs), desc="Pixel Cluster Mask Generation", unit="FOVs") \
            as pixel_mask_progress:
        for fov in fovs:
            pixel_mask_progress.set_postfix(FOV=fov)

            # define the path to provided channel file in the fov dir, used to calculate dimensions
            chan_file_path = os.path.join(fov, chan_file)

            # generate the pixel mask for the FOV
            pixel_mask: np.ndarray =\
                generate_pixel_cluster_mask(fov=fov, base_dir=base_dir, tiff_dir=tiff_dir,
                                            chan_file_path=chan_file_path,
                                            pixel_data_dir=pixel_data_dir,
                                            pixel_cluster_col=pixel_cluster_col,
                                            cluster_mapping=updated_cluster_map)

            # save the pixel mask generated
            save_fov_mask(fov, data_dir=save_dir, mask_data=pixel_mask, sub_dir=sub_dir,
                          name_suffix=name_suffix)

            pixel_mask_progress.update(1)


def generate_and_save_neighborhood_cluster_masks(
    fovs: List[str],
    save_dir: Union[pathlib.Path, str],
    seg_dir: Union[pathlib.Path, str],
    neighborhood_data: pd.DataFrame,
    fov_col: str = settings.FOV_ID,
    label_col: str = settings.CELL_LABEL,
    cluster_col: str = settings.KMEANS_CLUSTER,
    seg_suffix: str = "_whole_cell.tiff",
    xr_channel_name="label",
    sub_dir: Union[pathlib.Path, str] = None,
    name_suffix: str = "",
):
    """Generates neighborhood cluster masks and saves them for downstream analysis.

    Args:
        fovs (List[str]):
            A list of fovs to generate and save neighborhood masks for.
        save_dir (Union[pathlib.Path, str]):
            The directory to save the generated pixel cluster masks.
        seg_dir (Union[pathlib.Path, str]):
            The path to the segmentation data.
        neighborhood_data (pd.DataFrame):
            Contains the neighborhood cluster assignments for each cell.
        fov_col (str, optional):
            The column name containing the FOV IDs . Defaults to `settings.FOV_ID` (`"fov"`).
        label_col (str, optional):
            The column name containing the cell label. Defaults to `settings.CELL_LABEL`
            (`"label"`).
        cluster_col (str, optional):
            The column name containing the cluster label. Defaults to `settings.KMEANS_CLUSTER`
            (`"kmeans_neighborhood"`).
        seg_suffix (str, optional):
            The suffix that the segmentation images use. Defaults to `'_whole_cell.tiff'`
        xr_channel_name (str):
            Channel name for segmented data array.
        sub_dir (str, optional):
            The subdirectory to save the images in. If specified images are saved to
            `"data_dir/sub_dir"`. If `sub_dir = None` the images are saved to `"data_dir"`.
            Defaults to `None`.
        name_suffix (str, optional):
            Specify what to append at the end of every pixel mask. Defaults to `''`.
    """

    cmd = ClusterMaskData(
        data=neighborhood_data,
        fov_col=fov_col,
        label_col=label_col,
        cluster_col=cluster_col,
    )

    # create the neighborhood cluster masks across each fov
    with tqdm(total=len(fovs), desc="Neighborhood Cluster Mask Generation", unit="FOVs") \
            as neigh_mask_progress:
        # generate the mask for each FOV
        for fov in fovs:
            neigh_mask_progress.set_postfix(FOV=fov)

            # load in the label map for the FOV
            label_map = load_utils.load_imgs_from_dir(
                seg_dir,
                files=[fov + seg_suffix],
                xr_channel_names=[xr_channel_name],
                trim_suffix=seg_suffix.split(".")[0],
            ).loc[fov, ..., :]

            # generate the neighborhood mask for the FOV
            neighborhood_mask: np.ndarray = label_cells_by_cluster(fov, cmd, label_map)

            # save the neighborhood mask generated
            save_fov_mask(
                fov,
                data_dir=save_dir,
                mask_data=neighborhood_mask,
                sub_dir=sub_dir,
                name_suffix=name_suffix,
            )

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
    if img_sub_folder in [None, '', ""]:
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

    search_term: str = re.compile(r"(R\+?\d+)(C\+?\d+)")

    bad_fov_names = []
    for fov in fovs:
        res = re.search(search_term, fov)
        if res is None:
            bad_fov_names.append(fov)

    if len(bad_fov_names) > 0:
        raise ValueError(f"Invalid FOVs found in directory, {data_dir}. FOV names "
                         f"{bad_fov_names} should have the form RnCm.")

    # retrieve all extracted channel names and verify list if provided
    if not segmentation and not clustering:
        channel_imgs = io_utils.list_files(
            dir_name=os.path.join(data_dir, fovs[0], img_sub_folder),
            substrs=EXTENSION_TYPES["IMAGE"])
    else:
        channel_imgs = io_utils.list_files(data_dir, substrs=fovs[0]+'_')
        channel_imgs = [chan.split(fovs[0] + '_')[1] for chan in channel_imgs]

    if channels is None:
        channels = io_utils.remove_file_extensions(channel_imgs)
    else:
        misc_utils.verify_in_list(channel_inputs=channels,
                                  valid_channels=io_utils.remove_file_extensions(channel_imgs))

    file_ext = os.path.splitext(channel_imgs[0])[1]
    expected_tiles = load_utils.get_tiled_fov_names(fovs, return_dims=True)

    # save new images to the stitched_images, one channel at a time
    for chan, tile in itertools.product(channels, expected_tiles):
        prefix, expected_fovs, num_rows, num_cols = tile
        if prefix == "":
            prefix = "unnamed_tile"
        stitched_subdir = os.path.join(stitched_dir, prefix)
        if not os.path.exists(stitched_subdir):
            os.makedirs(stitched_subdir)
        image_data = load_utils.load_tiled_img_data(data_dir, fovs, expected_fovs, chan,
                                                    single_dir=any([segmentation, clustering]),
                                                    file_ext=file_ext[1:],
                                                    img_sub_folder=img_sub_folder)
        stitched_data = data_utils.stitch_images(image_data, num_cols)
        current_img = stitched_data.loc['stitched_image', :, :, chan].values
        image_utils.save_image(os.path.join(stitched_subdir, chan + '_stitched' + file_ext),
                               current_img)


def _convert_ct_fov_to_adata(fov_group: DataFrameGroupBy, var_names: list[str], obs_names: list[str], save_dir: os.PathLike) -> str:
    """Converts the cell table for a single FOV to an `AnnData` object and saves it to disk as a
    `Zarr` store.

    Parameters
    ----------
    fov_group : DataFrameGroupBy
        The cell table subset on a single FOV.
    var_names: list[str]
        The marker names to extract from the cell table.
    obs_names: list[str]
        The cell-level measurements and properties to extract from the cell table.
    save_dir: os.PathLike
        The directory to save the `AnnData` object to.

    Returns
    -------
    str
        The path of the saved `AnnData` object.
    """
    
    fov_pd: pd.DataFrame = fov_group.sort_values(by=settings.CELL_LABEL, key=ns.natsort_key).reset_index()
    fov_id: str = fov_pd[settings.FOV_ID].iloc[0]

    # Set the index to be the FOV and the segmentation label to create a unique index
    fov_pd.index = list(map(lambda label: f"{fov_id}_{int(label)}", fov_pd[settings.CELL_LABEL]))

    # Extract the X matrix
    X_dd: pd.DataFrame = fov_pd[var_names]
    
    # Extract the obs dataframe and convert the cell label to integer
    obs_pd: pd.DataFrame = fov_pd[obs_names].astype({settings.CELL_LABEL: int, settings.FOV_ID: str})
    obs_pd["cell_meta_cluster"] = pd.Categorical(obs_pd["cell_meta_cluster"].astype(str))

    # Move centroids from obs to obsm["spatial"]
    obsm_pd = obs_pd[[settings.CENTROID_0, settings.CENTROID_1]].rename(columns={settings.CENTROID_0: "centroid_y", settings.CENTROID_1: "centroid_x"})
    obs_pd = obs_pd.drop(columns=[settings.CENTROID_0, settings.CENTROID_1])

    # Create the AnnData object
    adata: AnnData = AnnData(X=X_dd, obs=obs_pd, obsm={"spatial": obsm_pd})

    # Convert any extra string labels to categorical if it's beneficial.
    adata.strings_to_categoricals()

    adata.write_zarr(pathlib.Path(save_dir, f"{fov_id}.zarr"), chunks=(1000, 1000))
    return pathlib.Path(save_dir, f"{fov_id}.zarr").as_posix()


class ConvertToAnnData:
    """ A class which converts the Cell Table `.csv` file to a series of `AnnData` objects,
    one object per FOV.
    
    The default parameters stored in the `.obs` slot include:
    - `area`
    - `cell_meta_cluster`
    - `centroid_dif`
    - `convex_area`
    - `convex_hull_resid`
    - `cell_meta_cluster`
    - `eccentricity`
    - `fov`
    - `major_axis_equiv_diam_ratio`

    Visit the Data Types document to see the full list of parameters.
    The default parameters stored in the `.obs` slot include:       
    - `centroid_x`
    - `centroid_y`

    Args:
        cell_table_path (os.PathLike): The path to the cell table.
        markers (list[str], "auto"): The markers to extract and store in `.X`. Defaults to "auto",
        which will extract all markers.
        extra_obs_parameters (list[str], optional): Extra parameters to load in `.obs`. Defaults
        to None.
    """

    def __init__(self, cell_table_path: os.PathLike,
                 markers: Union[list[str], Literal["auto"]] = "auto",
                 extra_obs_parameters: list[str] = None) -> None:
        
        io_utils.validate_paths(paths=cell_table_path)
        
        # Read in the cell table
        cell_table: pd.DataFrame = pd.read_csv(cell_table_path)
        ct_columns = cell_table.columns

        # Get the marker column indices
        marker_index_start: int = ct_columns.get_loc(settings.PRE_CHANNEL_COL) + 1
        marker_index_stop: int = ct_columns.get_loc(settings.POST_CHANNEL_COL)
        obs_index_start: int = ct_columns.get_loc(settings.POST_CHANNEL_COL) + 1
        
        if markers == "auto":
            # Default to all markers based on settings Pre and Post channel column values
            markers: list[str] = ct_columns[marker_index_start:marker_index_stop].to_list()
        else:
            # Verify that the correct markers exist
            misc_utils.verify_in_list(requested_markers=markers, 
                                    all_markers=ct_columns[marker_index_start:marker_index_stop].to_list())
        self.var_names = markers
        
        # Verify extra obs parameters
        if extra_obs_parameters:
            misc_utils.verify_in_list(requested_parameters=extra_obs_parameters, 
                                    all_parameters=ct_columns[obs_index_start:].to_list())
        else:
            extra_obs_parameters = []

        obs_names = [
            settings.CELL_LABEL,
            settings.CELL_SIZE,
            *ct_columns[obs_index_start:].to_list(),
            *extra_obs_parameters
        ]

        # Use "area" as the default area id instead of settings.CELL_SIZE to account for
        # non-cellular observations (ez_seg, fiber, etc...)
        if settings.CELL_SIZE in obs_names:
            obs_names.remove(settings.CELL_SIZE)
            if "area" not in obs_names:
                cell_table = cell_table.rename(columns={settings.CELL_SIZE: "area"})
                obs_names.append("area")
        
        self.obs_names: list[str] = obs_names
        self.cell_table = cell_table

    def convert_to_adata(
        self,
        save_dir: os.PathLike,
    ) -> dict[str, str]:
        """Converts the cell table to a FOV-level `AnnData` object, and saves the results as
        a `Zarr` store to disk in the `save_dir`.

        Args:
            save_dir (os.PathLike): The directory to save the `AnnData` objects to.

        Returns:
            dict[str, str]: A dictionary containing the names of the FOVs and the paths where
            they were saved.
        """

        if not isinstance(save_dir, pathlib.Path):
            save_dir = pathlib.Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        n_unique_fovs = self.cell_table[settings.FOV_ID].nunique()

        tqdm.pandas(desc="Converting Cell Table to AnnData Tables", total=n_unique_fovs, unit="FOVs")

        result: pd.Series = self.cell_table.groupby(by=settings.FOV_ID, sort=True).progress_apply(
            lambda x: _convert_ct_fov_to_adata(
                x, var_names=self.var_names, obs_names=self.obs_names, save_dir=save_dir
            ),
        )
        return result.to_dict()


class AnnCollectionKwargs(TypedDict):
    join_obs: Optional[Literal["inner", "outer"]]
    join_obsm: Optional[Literal["inner"]]
    join_vars: Optional[Literal["inner"]]
    label: Optional[str]
    keys: Optional[Sequence[str]]
    index_unique: Optional[str]
    convert: Optional[ConvertType]
    harmonize_dtypes: bool
    indices_strict: bool


def load_anndatas(anndata_dir: os.PathLike, **anncollection_kwargs: Unpack[AnnCollectionKwargs]) -> AnnCollection:
    """Lazily loads a directory of `AnnData` objects into an `AnnCollection`. The concatination happens across the `.obs` axis.
    
    For `AnnCollection` kwargs, see https://anndata.readthedocs.io/en/latest/generated/anndata.experimental.AnnCollection.html
        
    Args:
        anndata_dir (os.PathLike): The directory containing the `AnnData` objects.

    Returns:
        AnnCollection: The `AnnCollection` containing the `AnnData` objects.
    """
    if not isinstance(anndata_dir, pathlib.Path):
        anndata_dir = pathlib.Path(anndata_dir)
    
    adata_zarr_stores = {f.stem: read_zarr(f) for f in ns.natsorted(anndata_dir.glob("*.zarr"))}
    return AnnCollection(adatas=adata_zarr_stores, **anncollection_kwargs)


class AnnDataIterDataPipe(IterDataPipe):
    """The TorchData Iterable-style DataPipe. Takes an `AnnCollection`
    and makes it iterable by FOV for easy and flexible data pipelines.

    Args:
        fovs (AnnCollection): The `AnnCollection` containing the `AnnData` objects.
    """

    @property
    def fovs(self) -> AnnCollection:
        return self._fovs

    @fovs.setter
    def fovs(self, value: AnnCollection) -> None:
        self._fovs: AnnCollection = value

    def __init__(self, fovs: AnnCollection):
        self.fovs = fovs

    def __iter__(self) -> Iterator[AnnData]:
        yield from self.fovs.adatas
