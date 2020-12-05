import numpy as np
import os
import tempfile
from shutil import rmtree
import pytest

from ark.utils import data_utils, test_utils
import skimage.io as io


def test_generate_deepcell_input():
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs = ['fov1', 'fov2']
        chans = ['nuc1', 'nuc2', 'mem1', 'mem2']

        data_xr = test_utils.make_images_xarray(tif_data=None, fov_ids=fovs,
                                                channel_names=chans, dtype='int16')

        fov1path = os.path.join(temp_dir, 'fov1.tif')
        fov2path = os.path.join(temp_dir, 'fov2.tif')

        # test 1 nuc, 1 mem (no summing)
        nucs = ['nuc2']
        mems = ['mem2']

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = np.moveaxis(io.imread(fov1path), 0, -1)
        fov2 = np.moveaxis(io.imread(fov2path), 0, -1)

        assert np.array_equal(fov1, data_xr.loc['fov1', :, :, ['nuc2', 'mem2']].values)
        assert np.array_equal(fov2, data_xr.loc['fov2', :, :, ['nuc2', 'mem2']].values)

        # test 2 nuc, 2 mem (summing)
        nucs = ['nuc1', 'nuc2']
        mems = ['mem1', 'mem2']

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = np.moveaxis(io.imread(fov1path), 0, -1)
        fov2 = np.moveaxis(io.imread(fov2path), 0, -1)

        nuc_sums = data_xr.loc[:, :, :, nucs].sum(dim='channels').values
        mem_sums = data_xr.loc[:, :, :, mems].sum(dim='channels').values

        assert np.array_equal(fov1[:, :, 0], nuc_sums[0, :, :])
        assert np.array_equal(fov1[:, :, 1], mem_sums[0, :, :])
        assert np.array_equal(fov2[:, :, 0], nuc_sums[1, :, :])
        assert np.array_equal(fov2[:, :, 1], mem_sums[1, :, :])

        # test nuc None
        nucs = None

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = np.moveaxis(io.imread(fov1path), 0, -1)
        fov2 = np.moveaxis(io.imread(fov2path), 0, -1)

        assert np.all(fov1[:, :, 0] == 0)
        assert np.array_equal(fov1[:, :, 1], mem_sums[0, :, :])
        assert np.all(fov2[:, :, 0] == 0)
        assert np.array_equal(fov2[:, :, 1], mem_sums[1, :, :])

        # test mem None
        nucs = ['nuc2']
        mems = None

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = np.moveaxis(io.imread(fov1path), 0, -1)
        fov2 = np.moveaxis(io.imread(fov2path), 0, -1)

        assert np.all(fov1[:, :, 1] == 0)
        assert np.array_equal(fov1[:, :, 0], data_xr.loc['fov1', :, :, 'nuc2'].values)
        assert np.all(fov2[:, :, 1] == 0)
        assert np.array_equal(fov2[:, :, 0], data_xr.loc['fov2', :, :, 'nuc2'].values)

        # test nuc None and mem None
        with pytest.raises(ValueError):
            data_utils.generate_deepcell_input(data_xr, temp_dir, None, None)


def test_stitch_images():
    fovs, chans = test_utils.gen_fov_chan_names(num_fovs=40, num_chans=4)

    data_xr = test_utils.make_images_xarray(tif_data=None, fov_ids=fovs, channel_names=chans,
                                            dtype='int16')

    stitched_xr = data_utils.stitch_images(data_xr, 5)

    assert stitched_xr.shape == (1, 80, 50, 4)


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
