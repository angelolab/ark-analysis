import numpy as np
import os
import tempfile
from shutil import rmtree
import pytest
import pandas as pd
import xarray as xr

from ark.utils import data_utils, test_utils
import skimage.io as io

from ark.utils.data_utils import relabel_segmentation, label_cells_by_cluster


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


def test_relabel_segmentation():
    x = y = 5
    img_arr = np.arange(1, x * y + 1).reshape((x, y))
    d = {i: i + 1 for i in range(1, x * y + 1)}
    res = relabel_segmentation(img_arr, d)

    assert np.array_equal(img_arr + 1, res)

    # some cells are not mapped to any cluster-label
    d = {i: i + 1 for i in range(1, x * y - 5)}
    res = relabel_segmentation(img_arr, d)
    # these cells should all get a default label
    img_arr[img_arr >= x * y - 5] = x * y - 5

    assert np.array_equal(img_arr + 1, res)

    # test case for multiple pixels with the same label
    data = np.array([[1, 2], [3, 4]])
    data = np.repeat(data, 2)  # ([1, 1, 2, 2, 3, 3, 4, 4])
    img_arr = data.reshape((4, 2))
    d = {i: 10 * i for i in range(5)}
    res = relabel_segmentation(img_arr, d)

    assert np.array_equal(img_arr * 10, res)


def test_label_cells_by_cluster():
    fovs = ['fov1', 'fov2', 'fov3']
    x = y = 5
    cluster_labels = np.random.randint(1, 5, x * y * len(fovs))
    labels = [i % (x * y) for i in range(x * y * len(fovs))]
    data = zip(cluster_labels, labels, [fov for _ in range(x * y) for fov in fovs])
    all_data = pd.DataFrame(data, columns=['cluster_labels', 'label', 'fovs'])
    img_data = np.array([np.arange(1, x * y + 1).reshape((x, y)) for _ in fovs])

    # set random pixels to zero
    idx = np.random.choice(5, 3, replace=False)
    img_data[1][idx] = 0

    np.stack(img_data, axis=0)
    label_maps = xr.DataArray(img_data,
                              coords=[fovs, range(x), range(y)],
                              dims=["fovs", "rows", "cols"])
    res_xr = label_cells_by_cluster([fovs[0]], all_data, label_maps, fov_col='fovs')
    assert res_xr.shape == (1, x, y)

    res_xr = label_cells_by_cluster(fovs, all_data, label_maps, fov_col='fovs')
    assert res_xr.shape == (3, x, y)

    # zero pixels in fov1 should remain zero
    labeled_img = res_xr[res_xr['fovs'] == fovs[1]].values.squeeze()
    assert np.all(labeled_img[idx] == 0)

    # all pixels in fov2 should remain non-zero
    labeled_img = res_xr[res_xr['fovs'] == fovs[2]].values.squeeze()
    assert np.all(labeled_img[idx] > 0)
