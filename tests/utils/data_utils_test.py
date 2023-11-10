import os
import pathlib
import random
import shutil
import tempfile
from shutil import rmtree
from typing import Generator, Iterator, List, Tuple

import feather
import numpy as np
import pandas as pd
import pytest
import skimage.io as io
import xarray as xr
import numba as nb
import test_utils as ark_test_utils
from alpineer import image_utils, io_utils, load_utils, test_utils
from ark import settings
from ark.utils import data_utils

parametrize = pytest.mark.parametrize


@parametrize('sub_dir', [None, 'test_sub_dir'])
@parametrize('name_suffix', ['', 'test_name_suffix'])
def test_save_fov_mask(sub_dir, name_suffix):
    # define a sample FOV name
    fov = 'fov0'

    # generate sample image data
    sample_mask_data = np.random.randint(low=0, high=16, size=(40, 40), dtype='int16')

    # bad data_dir path provided
    with pytest.raises(FileNotFoundError):
        data_utils.save_fov_mask(fov, 'bad_data_path', sample_mask_data)

    with tempfile.TemporaryDirectory() as temp_dir:
        # test image saving
        data_utils.save_fov_mask(
            fov, temp_dir, sample_mask_data, sub_dir=sub_dir, name_suffix=name_suffix
        )

        # sub_dir gets set to empty string if left None
        if sub_dir is None:
            sub_dir = ''

        # assert the FOV file was created
        fov_img_path = os.path.join(temp_dir, sub_dir, fov + name_suffix + '.tiff')
        assert os.path.exists(fov_img_path)

        # load in the FOV file
        fov_img = io.imread(fov_img_path)

        # assert image was saved as np.int16
        assert fov_img.dtype == np.dtype('int16')

        # assert the image dimensions are correct
        assert fov_img.shape == (40, 40)


def test_erode_mask():
    seg_mask = np.zeros((10, 10))
    seg_mask[4:8, 4:8] = 1

    eroded_mask = data_utils.erode_mask(seg_mask)
    # (4 pixels left after the erosion step)
    assert eroded_mask[5:7, 5:7].sum() == 4


@pytest.fixture(scope="module")
def cell_table_cluster(rng: np.random.Generator) -> Generator[pd.DataFrame, None, None]:
    """_summary_

    Args:
        rng (np.random.Generator): _description_

    Yields:
        Generator[pd.DataFrame, None, None]: _description_
    """
    ct: pd.DataFrame = ark_test_utils.make_cell_table(num_cells=100)
    ct[settings.FOV_ID] = rng.choice(["fov0", "fov1"], size=100)
    ct["label"] = ct.groupby(by=settings.FOV_ID)["fov"].transform(
        lambda x: np.arange(start=1, stop=len(x) + 1, dtype=int)
    )
    ct[settings.CELL_TYPE] = rng.choice(["A", "B"], size=100)
    ct.reset_index(drop=True, inplace=True)
    yield ct


class TestClusterMaskData:
    @pytest.fixture(autouse=True)
    def _setup(self, cell_table_cluster: pd.DataFrame):
        self.cell_table: pd.DataFrame = cell_table_cluster
        self.label_column = "label"
        self.cluster_column = settings.CELL_TYPE
        self.fov_col = settings.FOV_ID

        self.cmd = data_utils.ClusterMaskData(
            data=self.cell_table,
            fov_col=self.fov_col,
            label_col=self.label_column,
            cluster_col=self.cluster_column,
        )

    def test___init__(self):
        # Test __init__
        assert self.cmd.label_column == self.label_column
        assert self.cmd.cluster_column == self.cluster_column
        assert self.cmd.fov_column == self.fov_col

        # Test __post_init__ generated fields
        assert set(self.cmd.unique_fovs) == set(["fov0", "fov1"])
        assert self.cmd.unassigned_id == 3
        assert isinstance(self.cmd.mapping, pd.DataFrame)

    @parametrize(
        "_fov", ["fov0", "fov1", pytest.param("fov2", marks=pytest.mark.xfail)]
    )
    def test_fov_mapping(self, _fov: str):
        fov_mapping_df: pd.DataFrame = self.cmd.fov_mapping(_fov)

        # The unassigned ID is the max label + 1, no label should be greater than this
        assert fov_mapping_df[self.cmd.cluster_id_column].max() <= self.cmd.unassigned_id

        # The background label is 0, no label should be less than this
        # And each FOV should have some background pixels
        assert fov_mapping_df["label"].min() == 0

    def test_cluster_ids(self):
        assert set(self.cmd.cluster_names) == {"A", "B"}


@pytest.fixture(scope="function")
def label_map_generator(
    cell_table_cluster: pd.DataFrame, rng: np.random.Generator
) -> Tuple[xr.DataArray, data_utils.ClusterMaskData]:
    """_summary_

    Args:
        cell_table_cluster (pd.DataFrame): The cell table with cluster assignments.
        rng (np.random.Generator): The random number generator.

    Yields:
        Tuple[xr.DataArray, data_utils.ClusterMaskData]: The label map and the ClusterMaskData.
    """
    fov_size = 40

    # Get the data for FOV 0
    fov0_data: pd.DataFrame = cell_table_cluster[
        cell_table_cluster[settings.FOV_ID] == "fov0"
    ]

    image_data: np.ndarray = rng.integers(
        low=0,
        high=fov0_data["label"].max() + 1,
        size=(fov_size, fov_size),
        dtype="int16",
    )

    label_map = xr.DataArray(
        data=image_data,
        coords=[np.arange(fov_size), np.arange(fov_size)],
        dims=["rows", "cols"],
    )

    cmd = data_utils.ClusterMaskData(
        data=cell_table_cluster,
        fov_col=settings.FOV_ID,
        label_col="label",
        cluster_col=settings.CELL_TYPE,
    )
    yield (label_map, cmd)


def test_label_cells_by_cluster(label_map_generator):
    label_map, cmd = label_map_generator

    relabeled_image: np.ndarray = data_utils.label_cells_by_cluster(
        fov="fov0", cmd=cmd, label_map=label_map)

    assert relabeled_image.max() <= cmd.unassigned_id
    assert relabeled_image.min() == 0
    assert relabeled_image.shape == label_map.shape


def test_map_segmentation_labels(
        rng: np.random.Generator,
        label_map_generator: Tuple[xr.DataArray, data_utils.ClusterMaskData]):

    data = pd.DataFrame(data={"labels": np.arange(start=1, stop=11), "values": np.concatenate(
        [[0], rng.random(size=8), [np.nan]])})

    label_map, _ = label_map_generator

    relabeled_image = data_utils.map_segmentation_labels(
        labels=data["labels"], values=data["values"], label_map=label_map)

    assert relabeled_image.max() <= data["values"].max()
    assert relabeled_image.min() == 0


def test_relabel_segmentation(label_map_generator):
    label_map, cmd = label_map_generator

    fov_clusters = cmd.fov_mapping("fov0")

    mapping: nb.typed.typeddict = nb.typed.Dict.empty(
        key_type=nb.types.int32,
        value_type=nb.types.int32,
    )

    for label, cluster in fov_clusters[[cmd.label_column, cmd.cluster_id_column]].itertuples(
            index=False):
        mapping[label] = cluster

    # Test the pure python counterpart alongside the numba version
    relabeled_image_py: np.ndarray = data_utils.relabel_segmentation.py_func(
        mapping=mapping,
        unassigned_id=cmd.unassigned_id,
        labeled_image=label_map.values,
    )
    relabeled_image_numba: np.ndarray = data_utils.relabel_segmentation(
        mapping=mapping,
        unassigned_id=cmd.unassigned_id,
        labeled_image=label_map.values,
    )
    for relabeled_image in [relabeled_image_py, relabeled_image_numba]:
        assert relabeled_image.max() <= cmd.unassigned_id
        assert relabeled_image.min() == 0
        assert relabeled_image.shape == label_map.shape


def test_generate_cluster_mask(tmp_path: pathlib.Path, label_map_generator):
    _, cmd = label_map_generator
    fov = 'fov0'
    som_cluster_cols = ['pixel_som_cluster_%d' % i for i in np.arange(5)]
    meta_cluster_cols = ['pixel_meta_cluster_%d' % i for i in np.arange(3)]

    with pytest.raises(ValueError):
        data_utils.generate_cluster_mask(
            fov, tmp_path, cmd, seg_suffix='bad_suffix'
        )

    # generate a sample segmentation mask
    cell_mask = np.random.randint(low=0, high=5, size=(40, 40), dtype="int16")
    image_utils.save_image(os.path.join(tmp_path, '%s_whole_cell.tiff' % fov), cell_mask)

    # create a sample cell consensus file based on SOM cluster assignments
    consensus_data_som = pd.DataFrame(
        np.random.randint(low=0, high=100, size=(20, 5)), columns=som_cluster_cols
    )

    consensus_data_som['fov'] = fov
    consensus_data_som['label'] = consensus_data_som.index.values + 1
    consensus_data_som['cell_som_cluster'] = np.tile(np.arange(1, 6), 4)
    consensus_data_som['cell_meta_cluster'] = np.tile(np.arange(1, 3), 10)

    # create a sample cell consensus file based on meta cluster assignments
    consensus_data_meta = pd.DataFrame(
        np.random.randint(low=0, high=100, size=(20, 3)), columns=meta_cluster_cols
    )

    consensus_data_meta['fov'] = fov
    consensus_data_meta['label'] = consensus_data_meta.index.values + 1
    consensus_data_meta['cell_som_cluster'] = np.tile(np.arange(1, 6), 4)
    consensus_data_meta['cell_meta_cluster'] = np.tile(np.arange(1, 3), 10)

    # bad cluster column provided
    with pytest.raises(ValueError):
        data_utils.generate_cluster_mask(
            fov="fov0", seg_dir=tmp_path, cmd=cmd, seg_suffix='bad_cluster'
        )

    # bad fov provided
    with pytest.raises(ValueError):
        data_utils.generate_cluster_mask(
            fov="fov1", seg_dir=tmp_path, cmd=cmd, seg_suffix="_whole_cell.tiff"
        )

    cell_masks = data_utils.generate_cluster_mask(
        fov, tmp_path, cmd, seg_suffix="_whole_cell.tiff"
    )

    # assert the image size is the same as the mask (40, 40)
    assert cell_masks.shape == (40, 40)

    # assert no value is greater than the highest SOM cluster value (5)
    assert np.all(cell_masks <= 5)


@parametrize('sub_dir', [None, 'sub_dir'])
@parametrize('name_suffix', ['', 'sample_suffix'])
def test_generate_and_save_cell_cluster_masks(tmp_path: pathlib.Path, sub_dir, name_suffix):
    fov_count = 7
    fovs = [f"fov{i}" for i in range(fov_count)]
    som_cluster_cols = ['pixel_som_cluster_%d' % i for i in np.arange(5)]
    meta_cluster_cols = ['pixel_meta_cluster_%d' % i for i in np.arange(3)]
    fov_size_split = 4

    # Create a save directory
    os.mkdir(os.path.join(tmp_path, 'cell_masks'))

    # generate sample segmentation masks
    # NOTE: this function should work on variable image sizes
    # TODO: condense cell mask generation into a helper function
    cell_masks_40 = np.random.randint(
        low=0, high=5, size=(fov_size_split, 40, 40, 1), dtype="int16"
    )

    cell_masks_20 = np.random.randint(
        low=0, high=5, size=(fov_count - fov_size_split, 20, 20, 1), dtype="int16"
    )

    for fov in range(fov_count):
        fov_index = fov if fov < fov_size_split else fov_size_split - fov
        fov_mask = cell_masks_40 if fov < fov_size_split else cell_masks_20
        fov_whole_cell = fov_mask[fov_index, :, :, 0]
        image_utils.save_image(os.path.join(tmp_path, 'fov%d_whole_cell.tiff' % fov),
                               fov_whole_cell)

    # create a sample cell consensus file based on SOM cluster assignments
    consensus_data_som = pd.DataFrame()

    # create a sample cell consensus file based on meta cluster assignments
    consensus_data_meta = pd.DataFrame()

    # generate sample cell data with SOM and meta cluster assignments for each fov
    for fov in fovs:
        som_data_fov = pd.DataFrame(
            np.random.randint(low=0, high=100, size=(20, 5)), columns=som_cluster_cols
        )

        som_data_fov['fov'] = fov
        som_data_fov['label'] = som_data_fov.index.values + 1
        som_data_fov['cell_som_cluster'] = np.tile(np.arange(1, 6), 4)
        som_data_fov['cell_meta_cluster'] = np.tile(np.arange(1, 3), 10)

        consensus_data_som = pd.concat([consensus_data_som, som_data_fov])

        meta_data_fov = pd.DataFrame(
            np.random.randint(low=0, high=100, size=(20, 3)), columns=meta_cluster_cols
        )

        meta_data_fov['fov'] = fov
        meta_data_fov['label'] = meta_data_fov.index.values + 1
        meta_data_fov['cell_som_cluster'] = np.tile(np.arange(1, 6), 4)
        meta_data_fov['cell_meta_cluster'] = np.tile(np.arange(1, 3), 10)

        consensus_data_meta = pd.concat([consensus_data_meta, meta_data_fov])

    # test various batch_sizes, no sub_dir, name_suffix = ''.
    data_utils.generate_and_save_cell_cluster_masks(
        fovs=fovs,
        save_dir=os.path.join(tmp_path, 'cell_masks'),
        seg_dir=tmp_path,
        cell_data=consensus_data_som,
        fov_col=settings.FOV_ID,
        label_col=settings.CELL_LABEL,
        cell_cluster_col='cell_som_cluster',
        seg_suffix='_whole_cell.tiff',
        sub_dir=sub_dir,
        name_suffix=name_suffix
    )

    # open each cell mask and make sure the shape and values are valid
    if sub_dir is None:
        sub_dir = ''

    for i, fov in enumerate(fovs):
        fov_name = fov + name_suffix + ".tiff"
        cell_mask = io.imread(os.path.join(tmp_path, 'cell_masks', sub_dir, fov_name))
        actual_img_dims = (40, 40) if i < fov_size_split else (20, 20)
        assert cell_mask.shape == actual_img_dims
        assert np.all(cell_mask <= 5)


def test_generate_pixel_cluster_mask():
    fov = 'fov0'
    chans = ['chan0', 'chan1', 'chan2', 'chan3']

    with tempfile.TemporaryDirectory() as temp_dir:
        # bad segmentation path passed
        with pytest.raises(FileNotFoundError):
            data_utils.generate_pixel_cluster_mask(
                fov, temp_dir, 'bad_tiff_dir', 'bad_chan_file', 'bad_consensus_path'
            )

        # bad channel file path passed
        with pytest.raises(FileNotFoundError):
            data_utils.generate_pixel_cluster_mask(
                fov, temp_dir, temp_dir, 'bad_chan_file', 'bad_consensus_path'
            )

        # generate sample fov folder with one channel value, no sub folder
        channel_data = np.random.randint(low=0, high=5, size=(40, 40), dtype="int16")
        os.mkdir(os.path.join(temp_dir, 'fov0'))
        image_utils.save_image(os.path.join(temp_dir, 'fov0', 'chan0.tiff'), channel_data)

        # bad consensus path passed
        with pytest.raises(FileNotFoundError):
            data_utils.generate_pixel_cluster_mask(
                fov, temp_dir, temp_dir, os.path.join('fov0', 'chan0.tiff'), 'bad_consensus_path'
            )

        # create a dummy consensus directory
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_consensus'))

        # create dummy data containing SOM and consensus labels for the fov
        consensus_data = pd.DataFrame(np.random.rand(100, 4), columns=chans)
        consensus_data['pixel_som_cluster'] = np.tile(np.arange(1, 11), 10)
        consensus_data['pixel_meta_cluster'] = np.tile(np.arange(1, 6), 20)
        consensus_data['row_index'] = np.random.randint(low=0, high=40, size=100)
        consensus_data['column_index'] = np.random.randint(low=0, high=40, size=100)

        feather.write_dataframe(
            consensus_data, os.path.join(temp_dir, 'pixel_mat_consensus', fov + '.feather')
        )

        # bad cluster column provided
        with pytest.raises(ValueError):
            data_utils.generate_pixel_cluster_mask(
                fov, temp_dir, temp_dir, os.path.join('fov0', 'chan0.tiff'),
                'pixel_mat_consensus', 'bad_cluster'
            )

        # bad fov provided
        with pytest.raises(ValueError):
            data_utils.generate_pixel_cluster_mask(
                'fov1', temp_dir, temp_dir, os.path.join('fov0', 'chan0.tiff'),
                'pixel_mat_consensus', 'pixel_som_cluster'
            )

        # test on SOM assignments
        pixel_masks = data_utils.generate_pixel_cluster_mask(
            fov, temp_dir, temp_dir, os.path.join('fov0', 'chan0.tiff'),
            'pixel_mat_consensus', 'pixel_som_cluster'
        )

        # assert the image size is the same as the mask (40, 40)
        assert pixel_masks.shape == (40, 40)

        # assert no value is greater than the highest SOM cluster value (10)
        assert np.all(pixel_masks <= 10)

        # test on meta assignments
        pixel_masks = data_utils.generate_pixel_cluster_mask(
            fov, temp_dir, temp_dir, os.path.join('fov0', 'chan0.tiff'),
            'pixel_mat_consensus', 'pixel_meta_cluster'
        )

        # assert the image size is the same as the mask (40, 40)
        assert pixel_masks.shape == (40, 40)

        # assert no value is greater than the highest meta cluster value (5)
        assert np.all(pixel_masks <= 5)


@parametrize('sub_dir', [None, 'sub_dir'])
@parametrize('name_suffix', ['', 'sample_suffix'])
def test_generate_and_save_pixel_cluster_masks(sub_dir, name_suffix):
    fov_count = 7
    fovs = [f"fov{i}" for i in range(fov_count)]
    chans = ['chan0', 'chan1', 'chan2', 'chan3']
    fov_size_split = 4

    with tempfile.TemporaryDirectory() as temp_dir:
        # create a dummy consensus directory
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_consensus'))

        # Create a save directory
        os.mkdir(os.path.join(temp_dir, 'pixel_masks'))

        # Name suffix
        name_suffix = ''

        # generate sample fov folders each with one channel value, no sub_dir
        # NOTE: this function should work on variable image sizes
        for i, fov in enumerate(fovs):
            chan_dims = (40, 40) if i < fov_size_split else (20, 20)
            channel_data = np.random.randint(low=0, high=5, size=chan_dims, dtype="int16")
            os.mkdir(os.path.join(temp_dir, fov))

            if not os.path.exists(os.path.join(temp_dir, fov)):
                os.mkdir(os.path.join(temp_dir, fov))

            image_utils.save_image(os.path.join(temp_dir, fov, 'chan0.tiff'), channel_data)

            consensus_data = pd.DataFrame(np.random.rand(100, 4), columns=chans)
            consensus_data['pixel_som_cluster'] = np.tile(np.arange(1, 11), 10)
            consensus_data['pixel_meta_cluster'] = np.tile(np.arange(1, 6), 20)
            consensus_data['row_index'] = np.random.randint(low=0, high=chan_dims[0], size=100)
            consensus_data['column_index'] = np.random.randint(low=0, high=chan_dims[1], size=100)

            feather.write_dataframe(
                consensus_data, os.path.join(temp_dir, 'pixel_mat_consensus', fov + '.feather')
            )

        data_utils.generate_and_save_pixel_cluster_masks(
            fovs=fovs,
            base_dir=temp_dir,
            save_dir=os.path.join(temp_dir, 'pixel_masks'),
            tiff_dir=temp_dir,
            chan_file='chan0.tiff',
            pixel_data_dir='pixel_mat_consensus',
            pixel_cluster_col='pixel_meta_cluster',
            sub_dir=sub_dir,
            name_suffix=name_suffix
        )

        # set sub_dir to empty string if None
        if sub_dir is None:
            sub_dir = ''

        # open each pixel mask and make sure the shape and values are valid
        for i, fov in enumerate(fovs):
            fov_name = fov + name_suffix + ".tiff"
            pixel_mask = io.imread(os.path.join(temp_dir, 'pixel_masks', sub_dir, fov_name))
            actual_img_dims = (40, 40) if i < fov_size_split else (20, 20)
            assert pixel_mask.shape == actual_img_dims
            assert np.all(pixel_mask <= 5)


@parametrize('sub_dir', [None, 'sub_dir'])
@parametrize('name_suffix', ['', 'sample_suffix'])
def test_generate_and_save_neighborhood_cluster_masks(sub_dir, name_suffix):
    fov_count = 5
    fovs = [f"fov{i}" for i in range(fov_count)]

    with tempfile.TemporaryDirectory() as temp_dir:
        # create a save directory
        os.mkdir(os.path.join(temp_dir, 'neighborhood_masks'))

        # create a segmentation dir
        os.mkdir(os.path.join(temp_dir, 'seg_dir'))

        # generate a neighborhood cluster DataFrame
        labels = np.arange(1, 6)
        sample_neighborhood_data = pd.DataFrame.from_dict(
            {settings.CELL_LABEL: np.repeat(labels, 5),
             settings.KMEANS_CLUSTER: np.repeat([i * 10 for i in labels], 5),
             settings.FOV_ID: np.tile(fovs, 5)}
        )

        # generate sample label map
        sample_label_maps = xr.DataArray(
            np.random.randint(low=0, high=5, size=(5, 40, 40), dtype="int16"),
            coords=[fovs, np.arange(40), np.arange(40)],
            dims=['fovs', 'rows', 'cols']
        )

        for fov in fovs:
            image_utils.save_image(
                os.path.join(temp_dir, 'seg_dir', fov + '_whole_cell.tiff'),
                sample_label_maps.loc[fov, ...].values,
            )

        data_utils.generate_and_save_neighborhood_cluster_masks(
            fovs=fovs,
            save_dir=os.path.join(temp_dir, 'neighborhood_masks'),
            seg_dir=os.path.join(temp_dir, 'seg_dir'),
            neighborhood_data=sample_neighborhood_data,
            fov_col=settings.FOV_ID,
            label_col=settings.CELL_LABEL,
            cluster_col=settings.KMEANS_CLUSTER,
            sub_dir=sub_dir,
            name_suffix=name_suffix
        )

        # set sub_dir to empty string if None
        if sub_dir is None:
            sub_dir = ''

        for i, fov in enumerate(fovs):
            fov_name = fov + name_suffix + ".tiff"
            neighborhood_mask = io.imread(
                os.path.join(temp_dir, 'neighborhood_masks', sub_dir, fov_name)
            )
            assert neighborhood_mask.shape == (40, 40)
            assert np.all(np.isin(neighborhood_mask, np.arange(6)))


def test_split_img_stack():
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs = ['stack_sample']
        _, chans, names = test_utils.gen_fov_chan_names(num_fovs=0, num_chans=10, return_imgs=True)

        stack_list = ["stack_sample.tiff"]
        stack_dir = os.path.join(temp_dir, fovs[0])
        os.mkdir(stack_dir)

        output_dir = os.path.join(temp_dir, "output_sample")
        os.mkdir(output_dir)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(stack_dir, fovs,
                                                                 chans, img_shape=(128, 128),
                                                                 mode='multitiff')

        # first test channel_first=False
        data_utils.split_img_stack(stack_dir, output_dir, stack_list, [0, 1], names[0:2],
                                   channels_first=False)

        assert os.path.exists(os.path.join(output_dir, "stack_sample", "chan0.tiff"))
        assert os.path.exists(os.path.join(output_dir, "stack_sample", "chan1.tiff"))

        sample_chan_1 = io.imread(os.path.join(output_dir, "stack_sample", "chan0.tiff"))
        sample_chan_2 = io.imread(os.path.join(output_dir, "stack_sample", "chan1.tiff"))

        assert np.array_equal(sample_chan_1, data_xr[0, :, :, 0].values)
        assert np.array_equal(sample_chan_2, data_xr[0, :, :, 1].values)

        rmtree(os.path.join(output_dir, 'stack_sample'))

        # now overwrite old stack_sample.jpg file and test channel_first=True
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(stack_dir, fovs,
                                                                 chans, img_shape=(128, 128),
                                                                 mode='reverse_multitiff')

        data_utils.split_img_stack(stack_dir, output_dir, stack_list, [0, 1], names[0:2],
                                   channels_first=True)

        assert os.path.exists(os.path.join(output_dir, "stack_sample", "chan0.tiff"))
        assert os.path.exists(os.path.join(output_dir, "stack_sample", "chan1.tiff"))

        sample_chan_1 = io.imread(os.path.join(output_dir, "stack_sample", "chan0.tiff"))
        sample_chan_2 = io.imread(os.path.join(output_dir, "stack_sample", "chan1.tiff"))

        assert np.array_equal(sample_chan_1, data_xr[0, :, :, 0].values)
        assert np.array_equal(sample_chan_2, data_xr[0, :, :, 1].values)


@pytest.fixture(scope="function", params=["no_prefix", "single_prefix", "multiple_prefixes"])
def stitching_fovs(request: str) -> Iterator[List[str]]:
    """
    A Fixture which yields a list of FOVs.

    Args:
        request (str): Either `no_prefix` or `run_prefix`. If it is `run_prefix` then the run
        will be prefixed with a random integer, `"run_i_RnCm"`. If it is `no_prefix`,
        then the FOV will be of the form `"RnCm"`

    Yields:
        Iterator[List[str]]: Returns a list of FOVs
    """
    param = request.param

    if param == "no_prefix":
        fovs: List[str] = [f"R{n}C{m}" for n in range(1, 14) for m in range(1, 14)]
    elif param == "single_prefix":
        fovs = [f"run_1_R{n}C{m}" for n in range(1, 14) for m in range(1, 14)]
    else:
        fovs = [f"run_1_R{n}C{m}" for n in range(1, 14) for m in range(1, 14)]
        fovs = fovs + [f"run_2_R{n}C{m}" for n in range(1, 14) for m in range(1, 14)]
    yield fovs


@pytest.mark.parametrize('segmentation, clustering, subdir',
                         [(False, False, 'TIFs'), (True, False, ''), (False, 'cell', ''),
                          (False, 'pixel', '')])
def test_stitch_images_by_shape(segmentation, clustering, subdir, stitching_fovs):

    # validation checks (only once)
    if clustering == 'pixel':
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, 'images')
            stitched_dir = os.path.join(temp_dir, 'stitched_images')
            os.makedirs(data_dir)

            # invalid directory is provided
            with pytest.raises(FileNotFoundError):
                data_utils.stitch_images_by_shape('not_a_dir', stitched_dir)

            # no fov dirs should raise an error
            with pytest.raises(ValueError, match="No FOVs found in directory"):
                data_utils.stitch_images_by_shape(data_dir, stitched_dir)

            for fov in ['fov1', 'fov2']:
                os.makedirs(os.path.join(data_dir, fov))

            # bad fov names should raise error
            with pytest.raises(ValueError, match="Invalid FOVs found in directory"):
                data_utils.stitch_images_by_shape(data_dir, stitched_dir)

            # one valid fov name but not all should raise error
            os.makedirs(os.path.join(temp_dir, 'R13C1'))
            with pytest.raises(ValueError, match="Invalid FOVs found in directory"):
                data_utils.stitch_images_by_shape(data_dir, stitched_dir)

            # bad clustering arg should raise an error
            with pytest.raises(ValueError, match="If stitching images from the pixie pipeline"):
                data_utils.stitch_images_by_shape(data_dir, stitched_dir,
                                                  segmentation=segmentation, clustering='not_cell')

            # check for existing previous stitched images
            os.makedirs(os.path.join(stitched_dir))
            with pytest.raises(ValueError, match="already exists"):
                data_utils.stitch_images_by_shape(data_dir, stitched_dir)

    # test success for various directory cases
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'images')
        stitched_dir = os.path.join(temp_dir, 'stitched_images')
        os.makedirs(data_dir)

        if segmentation:
            chans = ['nuclear', 'whole_cell']
        elif clustering:
            chans = [clustering + '_mask']
        else:
            chans = [f"chan{i}" for i in range(5)]
            # check that ignores toffy stitching in fov level dir
            stitching_fovs.append('stitched_images')

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            data_dir, stitching_fovs, chans,
            img_shape=(10, 10), fills=True, sub_dir=subdir, dtype=np.float32,
            single_dir=any([segmentation, clustering])
        )

        # bad channel name should raise an error
        with pytest.raises(ValueError, match="Not all values given in list"):
            data_utils.stitch_images_by_shape(data_dir, stitched_dir, channels='bad_channel',
                                              img_sub_folder=subdir, segmentation=segmentation,
                                              clustering=clustering)

        # test successful stitching
        if len(stitching_fovs) == 13*13*2:
            prefixes = ["run_1", "run_2"]
        elif stitching_fovs[0] == "R1C1":
            prefixes = ["unnamed_tile"]
        else:
            prefixes = ["run_1"]

        data_utils.stitch_images_by_shape(data_dir, stitched_dir, img_sub_folder=subdir,
                                          segmentation=segmentation, clustering=clustering)
        for prefix in prefixes:
            stitched_subdir = os.path.join(stitched_dir, prefix)
            assert sorted(io_utils.list_files(stitched_subdir)) == \
                [chan + '_stitched.tiff' for chan in chans]

            # stitched image is 13 x 13 fovs with max_img_size = 10, so the image is 130 x 130
            stitched_data = load_utils.load_imgs_from_dir(stitched_subdir,
                                                          files=[chans[0] + '_stitched.tiff'])
            assert stitched_data.shape == (1, 130, 130, 1)
        shutil.rmtree(stitched_dir)

        # test successful stitching for select channels
        random_channel = chans[random.randint(0, len(chans) - 1)]
        data_utils.stitch_images_by_shape(data_dir, stitched_dir, img_sub_folder=subdir,
                                          channels=[random_channel], segmentation=segmentation,
                                          clustering=clustering)
        for prefix in prefixes:
            stitched_subdir = os.path.join(stitched_dir, prefix)
            assert sorted(io_utils.list_files(stitched_subdir)) == \
                [random_channel + '_stitched.tiff']

        # remove stitched_images from fov list
        if not segmentation and not clustering:
            stitching_fovs.pop()
