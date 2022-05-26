import numpy as np
import pytest
import tempfile

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
                                                       delimiter='_',
                                                       dtype=np.float32)

        assert loaded_xr.equals(data_xr)

        # test delimiter agnosticism
        loaded_xr = load_utils.load_imgs_from_mibitiff(temp_dir,
                                                       mibitiff_files=fovnames,
                                                       channels=channels,
                                                       delimiter='_',
                                                       dtype=np.float32)

        assert loaded_xr.equals(data_xr)
        assert np.issubdtype(loaded_xr.dtype, np.floating)

        # test float overwrite
        with pytest.warns(UserWarning):
            loaded_xr = load_utils.load_imgs_from_mibitiff(temp_dir,
                                                           mibitiff_files=[fovnames[-1]],
                                                           channels=channels,
                                                           delimiter='_',
                                                           dtype='int16')

            assert loaded_xr.equals(data_xr.loc[[fovs[-1]], :, :, :])
            assert np.issubdtype(loaded_xr.dtype, np.floating)


def test_load_imgs_from_tree():
    # invalid directory is provided
    with pytest.raises(ValueError):
        loaded_xr = \
            load_utils.load_imgs_from_tree('not_a_dir', img_sub_folder="TIFs", dtype="int16")

    # test loading from within fov directories
    with tempfile.TemporaryDirectory() as temp_dir:
        # temp_dir contains no images
        with pytest.raises(ValueError):
            loaded_xr = \
                load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16")

        fovs, chans, imgs = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=3,
                                                          return_imgs=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, chans, img_shape=(10, 10), delimiter='_', fills=True, sub_dir="TIFs",
            dtype="int16"
        )

        with pytest.raises(ValueError):
            # attempt to pass an empty channels list
            loaded_xr = \
                load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs",
                                               dtype="int16", channels=[])

        # check default loading of all files
        loaded_xr = \
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16")

        assert loaded_xr.equals(data_xr)

        # check loading of specific files
        some_fovs = fovs[:2]
        some_imgs = imgs[:2]
        some_chans = chans[:2]

        loaded_xr = \
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           fovs=some_fovs, channels=some_imgs)

        assert loaded_xr.equals(data_xr[:2, :, :, :2])

        # check loading w/o file extension
        loaded_xr = \
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           channels=some_chans)

        assert loaded_xr.equals(data_xr[:, :, :, :2])

        # check mixed extension presence
        loaded_xr = \
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           channels=[chans[i] if i % 2 else imgs[i]
                                                     for i in range(3)])

        assert loaded_xr.equals(data_xr)

        # check when fov is a single string
        loaded_xr = \
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           fovs='fov0', channels=some_chans)

        assert loaded_xr.equals(data_xr[:1, :, :, :2])

        # check that an error raises when a channel provided does not exist
        with pytest.raises(ValueError):
            loaded_xr = \
                load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                               channels=['chan4'])

    # test loading with data_xr containing float values
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs, chans, imgs = test_utils.gen_fov_chan_names(num_fovs=1, num_chans=2,
                                                          return_imgs=True)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, chans, img_shape=(10, 10), delimiter='_', fills=True, sub_dir="TIFs",
            dtype=np.float32
        )

        with pytest.warns(UserWarning):
            loaded_xr = \
                load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16")

            assert loaded_xr.equals(data_xr)

            # test swap int16 -> float
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
            load_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           max_image_size=12)

        assert loaded_xr.shape == (3, 12, 12, 3)


def test_load_imgs_from_dir():
    # invalid directory is provided
    with pytest.raises(ValueError):
        loaded_xr = \
            load_utils.load_imgs_from_dir('not_a_dir', trim_suffix='_', dtype=np.float32)

    # test loading from 'free' directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # input directory contains no images
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, trim_suffix='_', dtype=np.float32)

        fovs, _ = test_utils.gen_fov_chan_names(num_fovs=3, num_chans=0, use_delimiter=True)
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(temp_dir, fovs, [0],
                                                                 img_shape=(10, 10), mode='labels',
                                                                 delimiter='_', dtype=np.float32)

        # invalid list of files is provided
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, files=fovs + ['not_an_image'],
                                          trim_suffix='_', dtype=np.float32)
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, files=['not_an_image'],
                                          trim_suffix='_', dtype=np.float32)

        # check default loading
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir, trim_suffix='_',
                                                  xr_dim_name='compartments', dtype=np.float32)

        assert loaded_xr.equals(data_xr)

        # check suffix matched loading:
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir, match_substring='_otherinfo',
                                                  trim_suffix='_', xr_dim_name='compartments',
                                                  dtype=np.float32)

        assert loaded_xr.equals(data_xr.loc[['fov0'], :, :, :])

        fovnames = [f'{fov}.tiff' for fov in fovs]

        # check general substring matched loading
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir, match_substring='ov', trim_suffix='_',
                                                  xr_dim_name='compartments', dtype=np.float32)

        assert loaded_xr.equals(data_xr)

        # check provided file overruling of match_substring
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir, files=fovnames,
                                                  match_substring='_otherinfo', trim_suffix='_',
                                                  xr_dim_name='compartments', dtype=np.float32)

        assert loaded_xr.equals(data_xr)

        # test error on no matched suffix
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, match_substring='not_a_real_suffix',
                                          trim_suffix='_', xr_dim_name='compartments',
                                          dtype=np.float32)

        # test swap float -> int16
        with pytest.warns(UserWarning):
            loaded_xr = load_utils.load_imgs_from_dir(temp_dir, trim_suffix='_', force_ints=True,
                                                      xr_dim_name='compartments', dtype="int16")

            assert loaded_xr.equals(data_xr)
            assert loaded_xr.dtype == 'int16'

        # test swap int16 -> float
        with pytest.warns(UserWarning):
            loaded_xr = load_utils.load_imgs_from_dir(temp_dir, trim_suffix='_',
                                                      xr_dim_name='compartments', dtype="int16")

            assert loaded_xr.equals(data_xr)
            assert np.issubdtype(loaded_xr.dtype, np.floating)

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
                                                  xr_dim_name='channels', trim_suffix='_',
                                                  dtype=np.float32)

        assert loaded_xr.equals(data_xr.loc[[fovs[-1]], :, :, :])

        # indices should be between 0-2
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, files=[fovnames[-1]], xr_dim_name='channels',
                                          trim_suffix='_', dtype=np.float32,
                                          channel_indices=[0, 1, 4])

        # xr_channel_names should contain 3 names (as there are 3 channels)
        with pytest.raises(ValueError):
            load_utils.load_imgs_from_dir(temp_dir, files=[fovnames[-1]], xr_dim_name='channels',
                                          trim_suffix='_', dtype=np.float32,
                                          xr_channel_names=['A', 'B'])

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
            fills=True, dtype=np.float32, channels_first=True
        )

        fovnames = [f'{fov}.tiff' for fov in fovs]

        # test all channels loading w/ specified file
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir, files=[fovnames[-1]],
                                                  xr_dim_name='channels', trim_suffix='_',
                                                  dtype=np.float32)

        assert loaded_xr.equals(data_xr.loc[[fovs[-1]], :, :, :])

        # test all channels w/ unspecified files + trim_suffix agnosticism
        loaded_xr = load_utils.load_imgs_from_dir(temp_dir,
                                                  files=None,
                                                  channel_indices=None,
                                                  xr_dim_name='channels',
                                                  trim_suffix='_')

        assert loaded_xr.equals(data_xr)
