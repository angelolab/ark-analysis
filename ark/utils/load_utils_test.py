import os
import pathlib
import shutil
import tempfile
from typing import Iterator, List, OrderedDict, Tuple

import numpy as np
import pytest
import xarray as xr
import xmltodict
from skimage import io
from tifffile import TiffFile, TiffWriter

from ark.utils import load_utils, test_utils


def test_load_imgs_from_mibitiff():
    # invalid directory is provided
    with pytest.raises(FileNotFoundError):
        loaded_xr = \
            load_utils.load_imgs_from_mibitiff('not_a_dir', channels=None, delimiter='_')

    with tempfile.TemporaryDirectory() as temp_dir:
        # temp_dir contains no images
        with pytest.raises(ValueError):
            loaded_xr = load_utils.load_imgs_from_mibitiff(temp_dir,
                                                           channels=None,
                                                           delimiter='_')

        # config test environment
        fovs, channels = test_utils.gen_fov_chan_names(num_fovs=2, num_chans=3, use_delimiter=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, channels, img_shape=(10, 10), mode='mibitiff', delimiter='_',
            fills=True, dtype=np.float32
        )

        with pytest.raises(ValueError):
            # attempt to pass an empty channels list
            loaded_xr = load_utils.load_imgs_from_mibitiff(temp_dir,
                                                           channels=[],
                                                           delimiter='_')

        # check unspecified fov loading
        loaded_xr = load_utils.load_imgs_from_mibitiff(temp_dir,
                                                       channels=channels,
                                                       delimiter='_')

        assert loaded_xr.equals(data_xr)

        fovnames = [f'{fov}.tiff' for fov in fovs]

        # check specified fov loading
        loaded_xr = load_utils.load_imgs_from_mibitiff(temp_dir,
                                                       mibitiff_files=[fovnames[-1]],
                                                       channels=channels,
                                                       delimiter='_')

        assert loaded_xr.equals(data_xr.loc[[fovs[-1]], :, :, :])

        # test automatic all channels loading
        loaded_xr = load_utils.load_imgs_from_mibitiff(temp_dir,
                                                       delimiter='_')

        assert loaded_xr.equals(data_xr)

        # test delimiter agnosticism
        loaded_xr = load_utils.load_imgs_from_mibitiff(temp_dir,
                                                       mibitiff_files=fovnames,
                                                       channels=channels,
                                                       delimiter='_')

        assert loaded_xr.equals(data_xr)
        assert np.issubdtype(loaded_xr.dtype, np.floating)


def test_load_imgs_from_tree():
    # invalid directory is provided
    with pytest.raises(FileNotFoundError):
        loaded_xr = \
            load_utils.load_imgs_from_tree('not_a_dir', img_sub_folder="TIFs")

    # test loading from within fov directories
    with tempfile.TemporaryDirectory() as temp_dir:
        # temp_dir contains no images
        with pytest.raises(ValueError):
            loaded_xr = \
                load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs")

        fovs, chans, imgs = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3,
                                                          return_imgs=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, chans, img_shape=(10, 10), delimiter='_', fills=True, sub_dir="TIFs",
            dtype="int16"
        )

        with pytest.raises(ValueError):
            # attempt to pass an empty channels list
            loaded_xr = \
                load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", channels=[])

        # check default loading of all files
        loaded_xr = \
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs")

        assert loaded_xr.equals(data_xr)

        # check loading of specific files
        some_fovs = fovs[:2]
        some_imgs = imgs[:2]
        some_chans = chans[:2]

        loaded_xr = \
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", fovs=some_fovs,
                                           channels=some_imgs)

        assert loaded_xr.equals(data_xr[:2, :, :, :2])

        # check loading w/o file extension
        loaded_xr = \
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", channels=some_chans)

        assert loaded_xr.equals(data_xr[:, :, :, :2])

        # check mixed extension presence
        loaded_xr = \
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs",
                                           channels=[chans[i] if i % 2 else imgs[i]
                                                     for i in range(3)])

        assert loaded_xr.equals(data_xr)

        # check when fov is a single string
        loaded_xr = \
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", fovs='fov0',
                                           channels=some_chans)

        assert loaded_xr.equals(data_xr[:1, :, :, :2])

        # check that an error raises when a channel provided does not exist
        with pytest.raises(ValueError):
            loaded_xr = \
                load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs",
                                               channels=['chan4'])

    # test loading with data_xr containing float values
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs, chans, imgs = test_utils.gen_fov_chan_names(num_fovs=1, num_chans=2,
                                                          return_imgs=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, chans, img_shape=(10, 10), delimiter='_', fills=True, sub_dir="TIFs",
            dtype=np.float32
        )

        loaded_xr = load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs")

        assert loaded_xr.equals(data_xr)
        assert np.issubdtype(loaded_xr.dtype, np.floating)

    # test loading with variable sizes
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs, chans, imgs = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3,
                                                          return_imgs=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, chans, img_shape=(10, 10), delimiter='_', fills=True, sub_dir="TIFs",
            dtype="int16"
        )

        loaded_xr = \
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", max_image_size=12)

        assert loaded_xr.shape == (3, 12, 12, 3)


def test_load_imgs_from_dir():
    # invalid directory is provided
    with pytest.raises(FileNotFoundError):
        loaded_xr = \
            load_utils.load_imgs_from_dir('not_a_dir', trim_suffix='_')

    # test loading from 'free' directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # input directory contains no images
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, trim_suffix='_')

        fovs, _ = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=0, use_delimiter=True)
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(temp_dir, fovs, [0],
                                                                 img_shape=(10, 10), mode='labels',
                                                                 delimiter='_', dtype=np.float32)

        # invalid list of files is provided
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, files=fovs + ['not_an_image'],
                                          trim_suffix='_')
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, files=['not_an_image'], trim_suffix='_')

        # check default loading
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir, trim_suffix='_',
                                                  xr_dim_name='compartments')

        assert loaded_xr.equals(data_xr)

        # check suffix matched loading:
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir, match_substring='_otherinfo',
                                                  trim_suffix='_', xr_dim_name='compartments')
        assert loaded_xr.equals(data_xr.loc[['fov0'], :, :, :])

        fovnames = [f'{fov}.tiff' for fov in fovs]

        # check general substring matched loading
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir, match_substring='ov', trim_suffix='_',
                                                  xr_dim_name='compartments')

        assert loaded_xr.equals(data_xr)

        # check provided file overruling of match_substring
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir, files=fovnames,
                                                  match_substring='_otherinfo', trim_suffix='_',
                                                  xr_dim_name='compartments')

        assert loaded_xr.equals(data_xr)

        # test error on no matched suffix
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, match_substring='not_a_real_suffix',
                                          trim_suffix='_', xr_dim_name='compartments')

    # Test floating point xarray type consistency and integer xarray type consistency
    # i.e. creation type == load_imgs type
    for dtype in [np.float32, np.int16]:
        with tempfile.TemporaryDirectory() as temp_dir:

            fovs, _ = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=0, use_delimiter=True)
            filelocs, data_xr = test_utils.create_paired_xarray_fovs(temp_dir, fovs, [0],
                                                                     img_shape=(10, 10),
                                                                     mode='labels',
                                                                     delimiter='_',
                                                                     dtype=dtype)

            # test to make sure that types stay consistent.
            loaded_xr = load_utils.load_imgs_from_dir(temp_dir, trim_suffix='_',
                                                      xr_dim_name='compartments')
            assert loaded_xr.equals(data_xr)
            assert loaded_xr.dtype == dtype

    # test multitiff input
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs, channels = test_utils.gen_fov_chan_names(num_fovs=2, num_chans=3, use_delimiter=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, channels, img_shape=(10, 10), mode='reverse_multitiff', delimiter='_',
            fills=True, dtype=np.float32
        )

        fovnames = [f'{fov}.tiff' for fov in fovs]

        # test all channels loading w/ specified file
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir, files=[fovnames[-1]],
                                                  xr_dim_name='channels', trim_suffix='_')
        assert loaded_xr.equals(data_xr.loc[[fovs[-1]], :, :, :])

        # indices should be between 0-2
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, files=[fovnames[-1]], xr_dim_name='channels',
                                          trim_suffix='_', channel_indices=[0, 1, 4])

        # xr_channel_names should contain 3 names (as there are 3 channels)
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, files=[fovnames[-1]], xr_dim_name='channels',
                                          trim_suffix='_', xr_channel_names=['A', 'B'])

        # test all channels w/ unspecified files + trim_suffix agnosticism
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir,
                                                  files=None,
                                                  channel_indices=None,
                                                  xr_dim_name='channels',
                                                  trim_suffix='_')

        assert loaded_xr.equals(data_xr)

        # test with specified channel_indices
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir,
                                                  files=None,
                                                  channel_indices=[0, 1, 2],
                                                  xr_dim_name='channels',
                                                  trim_suffix='_')

        assert loaded_xr.equals(data_xr[:, :, :, :3])

        # test channels_first input
        fovs, channels = test_utils.gen_fov_chan_names(num_fovs=2, num_chans=5, use_delimiter=True)

        _, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, channels, img_shape=(10, 10), mode='multitiff', delimiter='_',
            fills=True, channels_first=True
        )

        fovnames = [f'{fov}.tiff' for fov in fovs]

        # test all channels loading w/ specified file
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir, files=[fovnames[-1]],
                                                  xr_dim_name='channels', trim_suffix='_')
        assert loaded_xr.equals(data_xr.loc[[fovs[-1]], :, :, :])

        # test all channels w/ unspecified files + trim_suffix agnosticism
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir,
                                                  files=None,
                                                  channel_indices=None,
                                                  xr_dim_name='channels',
                                                  trim_suffix='_')

        assert loaded_xr.equals(data_xr)


def test_check_fov_name_prefix():
    # check no prefix
    prefix, fovs = load_utils.check_fov_name_prefix(['R1C1', 'R1C2', 'R1C3'])
    assert prefix is False and fovs == ['R1C1', 'R1C2', 'R1C3']

    # check all fovs have prefix
    prefix, fovs = load_utils.check_fov_name_prefix(['Run_1_R1C1', 'Run_2_R1C2', 'Run_1_R1C3'])
    assert prefix is True and fovs == {'R1C1': 'Run_1', 'R1C2': 'Run_2', 'R1C3': 'Run_1'}

    # check some fovs have prefix
    prefix, fovs = load_utils.check_fov_name_prefix(['R1C1', 'R1C2', 'run1_R1C3'])
    assert prefix is True and fovs == {'R1C1': '', 'R1C2': '', 'R1C3': 'run1'}


def test_get_tiled_fov_names():
    # check no missing fovs, should return a list with all fovs for a 3x4 tiling
    fov_names = ['R1C1', 'R1C2', 'R2C1', 'R2C2']

    expected_fovs = load_utils.get_tiled_fov_names(fov_names)
    assert expected_fovs == ['R1C1', 'R1C2', 'R2C1', 'R2C2']

    # check no missing fovs and run name attached, should return a list for 1x3 tiling
    fov_names = ['Run_10_R1C1', 'Run_10_R1C2', 'Run_20_R1C3']

    expected_fovs, rows, cols = load_utils.get_tiled_fov_names(fov_names, return_dims=True)
    assert expected_fovs == ['Run_10_R1C1', 'Run_10_R1C2', 'Run_20_R1C3']
    assert (rows, cols) == (1, 3)

    # check missing fovs, should return a list with all fovs for a 3x4 tiling
    fov_names = ['R1C1', 'R1C2', 'R2C1', 'R2C4', 'RC3C1']

    expected_fovs, rows, cols = load_utils.get_tiled_fov_names(fov_names, return_dims=True)
    assert expected_fovs == ['R1C1', 'R1C2', 'R1C3', 'R1C4', 'R2C1', 'R2C2', 'R2C3', 'R2C4',
                             'R3C1', 'R3C2', 'R3C3', 'R3C4']
    assert (rows, cols) == (3, 4)

    # check missing fovs with run name attached, should return a list with all fovs for 1x3 tiling
    fov_names = ['Run_10_R1C1', 'Run_20_R1C3']

    expected_fovs, rows, cols = load_utils.get_tiled_fov_names(fov_names, return_dims=True)
    assert expected_fovs == ['Run_10_R1C1', 'R1C2', 'Run_20_R1C3']
    assert (rows, cols) == (1, 3)


@pytest.mark.parametrize('single_dir, img_sub_folder', [(False, 'TIFs'), (True, '')])
def test_load_tiled_img_data(single_dir, img_sub_folder):
    # invalid directory is provided
    with pytest.raises(FileNotFoundError):
        loaded_xr = load_utils.load_tiled_img_data('not_a_dir', [], [], 'chan1',
                                                   single_dir=False,)

    # check with no missing FOVS
    with tempfile.TemporaryDirectory() as temp_dir:

        fovs = ['R1C1', 'R1C2', 'R1C3']
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, ['chan1', 'chan2'], img_shape=(10, 10), fills=True,
            sub_dir=img_sub_folder, dtype="int16", single_dir=single_dir
        )

        # check default loading of chan1 images
        loaded_xr = load_utils.load_tiled_img_data(temp_dir, fovs, expected_fovs=fovs,
                                                   channel='chan1', single_dir=single_dir,
                                                   img_sub_folder=img_sub_folder)

        assert loaded_xr.equals(data_xr[:, :, :, :-1])
        assert loaded_xr.shape == (3, 10, 10, 1)

        # check toffy dict loading
        if not single_dir:
            toffy_fovs = {'R1C1': 'fov-1', 'R1C2': 'fov-3', 'R1C3': 'fov-2'}
            fovs = list(toffy_fovs.values())
            expected_fovs = load_utils.get_tiled_fov_names(list(toffy_fovs.keys()))

            filelocs, data_xr = test_utils.create_paired_xarray_fovs(
                temp_dir, fovs, ['chan1', 'chan2'], img_shape=(10, 10), fills=True,
                sub_dir=img_sub_folder, dtype="int16", single_dir=single_dir
            )

            # check default loading of chan1 images
            loaded_xr = load_utils.load_tiled_img_data(temp_dir, toffy_fovs,
                                                       expected_fovs=expected_fovs,
                                                       channel='chan1', single_dir=single_dir,
                                                       img_sub_folder=img_sub_folder)

            assert loaded_xr.equals(data_xr[:, :, :, :-1])
            assert loaded_xr.shape == (3, 10, 10, 1)

    # check loading with missing FOV images
    with tempfile.TemporaryDirectory() as temp_dir:

        fovs = ['R1C1', 'R1C2', 'R2C1', 'R2C2']
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, ['chan1', 'chan2'], img_shape=(10, 10), delimiter='_', fills=True,
            sub_dir=img_sub_folder, dtype="int16", single_dir=single_dir
        )
        # remove images and expected data for one fov
        data_xr[2, :, :, :] = np.zeros((10, 10, 1), dtype='int16')
        if single_dir:
            os.remove(os.path.join(temp_dir, 'R2C1_chan1.tiff'))
        else:
            shutil.rmtree(os.path.join(temp_dir, 'R2C1'))

        # check successful loading for one channel
        loaded_xr = \
            load_utils.load_tiled_img_data(temp_dir, ['R1C1', 'R1C2', 'R2C2'], fovs, 'chan1',
                                           single_dir=single_dir, img_sub_folder=img_sub_folder)
        assert loaded_xr.equals(data_xr[:, :, :, :-1])
        assert loaded_xr.shape == (4, 10, 10, 1)

        # check toffy dict loading
        if not single_dir:
            toffy_fovs = {'R1C1': 'fov-3', 'R1C2': 'fov-1', 'R2C1': 'fov-4', 'R2C2': 'fov-2'}
            fovs = list(toffy_fovs.values())
            expected_fovs = load_utils.get_tiled_fov_names(list(toffy_fovs.keys()))

            filelocs, data_xr = test_utils.create_paired_xarray_fovs(
                temp_dir, fovs, ['chan1', 'chan2'], img_shape=(10, 10), delimiter='_', fills=True,
                sub_dir=img_sub_folder, dtype="int16", single_dir=single_dir
            )
            data_xr['fovs'] = list(toffy_fovs.keys())

            # remove images and expected data for one fov
            data_xr[2, :, :, :] = np.zeros((10, 10, 1), dtype='int16')
            shutil.rmtree(os.path.join(temp_dir, 'fov-4'))
            toffy_fovs.pop('R2C1')

            # check successful loading for one channel
            loaded_xr = \
                load_utils.load_tiled_img_data(temp_dir, toffy_fovs, expected_fovs,
                                               'chan1', single_dir=single_dir,
                                               img_sub_folder=img_sub_folder)

            assert loaded_xr.equals(data_xr[:, :, :, :-1])
            assert loaded_xr.shape == (4, 10, 10, 1)

    # test loading with data_xr containing float values
    with tempfile.TemporaryDirectory() as temp_dir:

        fovs = ['R1C1', 'R1C2', 'R2C1', 'R2C2']
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, ['chan1'], img_shape=(10, 10), delimiter='_', fills=True,
            sub_dir=img_sub_folder, dtype=np.float32, single_dir=single_dir
        )
        # remove images and expected data for one fov
        data_xr[2, :, :, :] = np.zeros((10, 10, 1), dtype='int16')
        if single_dir:
            os.remove(os.path.join(temp_dir, 'R2C1_chan1.tiff'))
        else:
            shutil.rmtree(os.path.join(temp_dir, 'R2C1'))

        loaded_xr = load_utils.load_tiled_img_data(temp_dir, ['R1C1', 'R1C2', 'R2C2'], fovs,
                                                   'chan1', single_dir=single_dir,
                                                   img_sub_folder=img_sub_folder)
        assert loaded_xr.shape == (4, 10, 10, 1)
        assert np.issubdtype(loaded_xr.dtype, np.floating)

    # test loading with run name prepend
    with tempfile.TemporaryDirectory() as temp_dir:

        fovs = ['run_1_R1C1', 'run_1_R1C2', 'R2C1', 'run_2_R2C2']
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, ['chan1'], img_shape=(10, 10), fills=True,
            sub_dir=img_sub_folder, dtype="int16", single_dir=single_dir
        )
        # remove images and expected data for one fov
        data_xr[2, :, :, :] = np.zeros((10, 10, 1), dtype='int16')
        if single_dir:
            os.remove(os.path.join(temp_dir, 'R2C1_chan1.tiff'))
        else:
            shutil.rmtree(os.path.join(temp_dir, 'R2C1'))

        loaded_xr = \
            load_utils.load_tiled_img_data(temp_dir, ['run_1_R1C1', 'run_1_R1C2', 'run_2_R2C2'],
                                           fovs, 'chan1', single_dir=single_dir,
                                           img_sub_folder=img_sub_folder)
        assert loaded_xr.equals(data_xr)
        assert loaded_xr.shape == (4, 10, 10, 1)


@pytest.fixture(scope="module")
def create_img_data(tmp_path_factory) -> Iterator[Tuple[pathlib.Path, xr.DataArray]]:
    """
    A Fixture which creates a temporary directory, FOVs for testing, and OME-TIFFs for testing.

    Args:
        tmp_path_factory (pytest.TempPathFactory): Factory for temporary directories under the
            common base temp directory.

    Yields:
        Iterator[Tuple[pathlib.Path, xr.DataArray]]: A tuple of the temporary directory path
        which contains the FOVs and OME-TIFFs and the xarray containing the image data.
    """

    # Create a temporary directory for the FOVs
    test_dir = tmp_path_factory.mktemp("ome_tests")
    # Create 3 FOVs, 10 channels each
    _, data_xr = test_utils.create_paired_xarray_fovs(
        base_dir=test_dir,
        fov_names=[f"fov{i}" for i in range(3)],
        channel_names=[f"chan{i}" for i in range(10)],
        mode='tiff', fills=False
        )

    # Create OME-TIFFs
    _compression: dict = {
        "algorithm": "zlib",
        "args": {"level": 6}
    }
    data_xr_transposed = data_xr.transpose("fovs", "channels", "cols", "rows")
    for fov in data_xr_transposed:
        fov_name: str = fov.fovs.values
        ome_file_path: pathlib.Path = pathlib.Path(test_dir) / f"{fov_name}.ome.tiff"

        _metadata = {
            "axes": "CYX",
            "Channel": {"Name": fov.channels.values.tolist()},
            "Name": fov_name
        }

        with TiffWriter(ome_file_path, ome=True) as ome_tiff:
            ome_tiff.write(
                data=fov.values,
                photometric="minisblack",
                compression=_compression["algorithm"],
                compressionargs=_compression["args"],
                metadata=_metadata
            )

    yield (test_dir, data_xr)


class TestOMEConversion:
    @pytest.fixture(autouse=True)
    def _setup(self, create_img_data, tmp_path):
        self.test_dir, self.data_xr = create_img_data
        self.save_path = tmp_path / "conversion_dir"
        self.save_path.mkdir()

    @pytest.mark.parametrize("fovs", [["fov0"], ["fov0", "fov1"], None],
                             ids=["single_fov", "multiple_fovs", "all_fovs"])
    @pytest.mark.parametrize("channels", [["chan0"], ["chan0", "chan1", "chan3"], None],
                             ids=["single_channel", "multiple_channels", "all_channels"])
    def test_fov_to_ome(self, fovs, channels) -> None:

        load_utils.fov_to_ome(data_dir=self.test_dir, ome_save_dir=self.save_path,
                              fovs=fovs, channels=channels)

        # Gather the correct FOV names
        fovs = fovs if fovs else self.data_xr.fovs.values
        channels = channels if channels else self.data_xr.channels.values

        # Assert that the OME-TIFF filenames exist and are correct
        for fov_name in fovs:
            assert os.path.exists(self.save_path / f"{fov_name}.ome.tiff")

        # Assert that the OME-TIFF files contain the correct data / metadata
        for fov_name in fovs:
            with TiffFile(self.save_path / f"{fov_name}.ome.tiff") as ome_tiff:
                _image_name, _channels = self._get_ome_metadata(ome_tiff)

                # Assert that the image name in the metadata is correct
                assert _image_name == fov_name

                # Assert that the channel values in the OME-TIFF are correct.
                for page, chan in zip(ome_tiff.pages, channels):
                    actual_data = page.asarray().transpose()
                    desired_data = self.data_xr.sel(fovs=fov_name, channels=chan).values
                    np.testing.assert_equal(actual_data, desired_data)

    def test_ome_to_fov(self) -> None:
        # Only need to test for 1 FOV / 1 OME-TIFF b/c `ome_to_fov` is `1-to-1'
        load_utils.ome_to_fov(ome=self.test_dir / "fov0.ome.tiff", data_dir=self.save_path)

        # Assert that the FOV directory exists
        os.path.exists(self.save_path / "fov0")

        # Assert that the FOV channels exist
        for chan in self.data_xr.channels.values:
            assert os.path.exists(self.save_path / "fov0" / f"{chan}.tiff")

        # Assert that the channel values are correct
        for chan in self.data_xr.channels.values:
            # Read in the channel data
            actual_data: np.ndarray = io.imread(self.save_path / "fov0" / f"{chan}.tiff")
            desired_data: np.ndarray = self.data_xr.sel(fovs="fov0", channels=chan).values
            np.testing.assert_equal(actual_data, desired_data)

    def _get_ome_metadata(self, ome_tiff: TiffFile) -> Tuple[str, List[str]]:
        ome_xml_metadata = xmltodict.parse(ome_tiff.ome_metadata)
        image_name = ome_xml_metadata["OME"]["Image"]["@Name"]
        channel_metadata: OrderedDict = ome_xml_metadata["OME"]["Image"]["Pixels"]["Channel"]

        if isinstance(channel_metadata, dict):
            channel_metadata = [channel_metadata]
        channels = list(map(lambda x: x["@Name"], channel_metadata))

        return (image_name, channels)
