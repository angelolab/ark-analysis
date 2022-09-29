import os
import tempfile
import shutil

import numpy as np
import pytest

from ark.utils import load_utils, test_utils


def test_load_imgs_from_mibitiff():
    # invalid directory is provided
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
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
    # check no missing fovs
    fov_names = ['R1C1', 'R1C2', 'R2C1', 'R2C2']
    # should return a list with all fovs for a 3x4 tiled image
    expected_fovs = load_utils.get_tiled_fov_names(fov_names)
    assert expected_fovs == ['R1C1', 'R1C2', 'R2C1', 'R2C2']

    # check missing fovs
    fov_names = ['R1C1', 'R1C2', 'R2C1', 'R2C4', 'RC3C1']

    # should return a list with all fovs for a 3x4 tiled image
    expected_fovs, rows, cols = load_utils.get_tiled_fov_names(fov_names, return_dims=True)
    assert expected_fovs == ['R1C1', 'R1C2', 'R1C3', 'R1C4', 'R2C1', 'R2C2', 'R2C3', 'R2C4',
                             'R3C1', 'R3C2', 'R3C3', 'R3C4']
    assert (rows, cols) == (3, 4)

    # check missing fovs with run name attached
    fov_names = ['Run_10_R1C1', 'Run_20_R1C3']

    # should return a list with all fovs for a 3x4 tiled image
    expected_fovs, rows, cols = load_utils.get_tiled_fov_names(fov_names, return_dims=True)
    assert expected_fovs == ['Run_10_R1C1', 'R1C2', 'Run_20_R1C3']
    assert (rows, cols) == (1, 3)


def test_get_max_img_size():
    with tempfile.TemporaryDirectory() as tmpdir:
        channel_list = ['Au', 'CD3', 'CD4', 'CD8', 'CD11c']
        fov_list = ['fov-1-scan-1', 'fov-2-scan-1']
        stitched_dir = ['stitched_images']
        larger_fov = ['fov-3-scan-1']

        test_utils._write_tifs(tmpdir, fov_list, channel_list, (10, 10), '', False, int)
        test_utils._write_tifs(tmpdir, stitched_dir, channel_list, (12, 12), '', False, int)

        # test success excluding stitched dir
        max_img_size = load_utils.get_max_img_size(tmpdir)
        assert max_img_size == 10

        test_utils._write_tifs(tmpdir, larger_fov, channel_list, (12, 12), '', False, int)

        # test success for all fovs
        max_img_size = load_utils.get_max_img_size(tmpdir)
        assert max_img_size == 12

        # write images to subfolder
        test_utils._write_tifs(tmpdir, fov_list, channel_list, (10, 10), 'TIFs', False, int)
        test_utils._write_tifs(tmpdir, larger_fov, channel_list, (12, 12), 'TIFs', False, int)

        # test success with subfolder
        max_img_size = load_utils.get_max_img_size(tmpdir, img_sub_folder='TIFs')
        assert max_img_size == 12

    with tempfile.TemporaryDirectory() as tmpdir:
        fovs = ['fov1_feature_0', 'fov1_feature_1', 'fov2_feature_0', 'fov2_feature_1']
        larger_fov = ['fov3_feature_0', 'fov3_feature_1']

        test_utils._write_tifs(tmpdir, ['deepcell_output'], fovs, (10, 10), '', False, int)
        test_utils._write_tifs(tmpdir, ['temp_fov_dir'], larger_fov, (12, 12), '', False, int)

        for img in larger_fov:
            shutil.copy(os.path.join(tmpdir, 'temp_fov_dir', img + '.tiff'),
                        os.path.join(tmpdir, 'deepcell_output', img + '.tiff'))

        # test success for all fovs in single dir
        max_img_size = load_utils.get_max_img_size(os.path.join(tmpdir, 'deepcell_output'),
                                                   single_dir=True)
        assert max_img_size == 12


@pytest.mark.parametrize('single_dir', [False, True])
def test_load_tiled_img_data(single_dir):
    # invalid directory is provided
    with pytest.raises(ValueError):
        loaded_xr = load_utils.load_tiled_img_data('not_a_dir', [], 'chan1', max_image_size=10,
                                                   single_dir=False,)
    if single_dir:
        img_sub_folder = ''
    else:
        img_sub_folder = 'TIFs'

    # check with no missing FOVS
    with tempfile.TemporaryDirectory() as temp_dir:

        fovs = ['R1C1', 'R1C2', 'R1C3']
        channels = ['chan1', 'chan2']

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, channels, img_shape=(10, 10), delimiter='_', fills=True,
            sub_dir=img_sub_folder, dtype="int16", single_dir=single_dir
        )

        # check default loading of chan1 images
        loaded_xr = load_utils.load_tiled_img_data(temp_dir, fovs, 'chan1', max_image_size=10,
                                                   single_dir=single_dir,
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
        # missing fov data
        data_xr[2, :, :, :] = np.zeros((10, 10, 1), dtype='int16')
        if single_dir:
            os.remove(os.path.join(temp_dir, 'R2C1_chan1.tiff'))
        else:
            shutil.rmtree(os.path.join(temp_dir, 'R2C1'))

        # check successful loading for one channel
        loaded_xr = \
            load_utils.load_tiled_img_data(temp_dir, ['R1C1', 'R1C2', 'R2C2'], 'chan1',
                                           max_image_size=10, single_dir=single_dir,
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
        # missing fov data
        data_xr[2, :, :, :] = np.zeros((10, 10, 1), dtype='int16')
        if single_dir:
            os.remove(os.path.join(temp_dir, 'R2C1_chan1.tiff'))
        else:
            shutil.rmtree(os.path.join(temp_dir, 'R2C1'))

        loaded_xr = load_utils.load_tiled_img_data(temp_dir, ['R1C1', 'R1C2', 'R2C2'], 'chan1',
                                                   max_image_size=10, single_dir=single_dir,
                                                   img_sub_folder=img_sub_folder)

        assert loaded_xr.equals(data_xr)
        assert np.issubdtype(loaded_xr.dtype, np.floating)

    # test loading with run name prepend and image padding
    with tempfile.TemporaryDirectory() as temp_dir:

        fovs = ['run_1_R1C1', 'run_1_R1C2', 'run_2_R2C1', 'run_2_R2C2']
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, ['chan1'], img_shape=(10, 10), delimiter='_', fills=True,
            sub_dir=img_sub_folder, dtype="int16", single_dir=single_dir
        )
        
        # missing fov data
        data_xr[2, :, :, :] = np.zeros((10, 10, 1), dtype='int16')
        if single_dir:
            os.remove(os.path.join(temp_dir, 'run_2_R2C1_chan1.tiff'))
        else:
            shutil.rmtree(os.path.join(temp_dir, 'run_2_R2C1'))

        loaded_xr = \
            load_utils.load_tiled_img_data(temp_dir, ['run_1_R1C1', 'run_1_R1C2', 'run_2_R2C2'],
                                           'chan1', max_image_size=12, single_dir=single_dir,
                                           img_sub_folder=img_sub_folder)
        assert loaded_xr.shape == (4, 12, 12, 1)
