import numpy as np
import os
import tempfile
from shutil import rmtree
import pytest
import feather
import pandas as pd
import xarray as xr

from ark.utils import data_utils, test_utils
import skimage.io as io

from ark.utils.data_utils import relabel_segmentation, label_cells_by_cluster
from ark import settings


def test_save_fov_images():
    # define the full set of fovs as well as a subset of fovs
    fovs = ['fov0', 'fov1', 'fov2']
    fovs_sub = fovs[:2]

    # generate a random img_xr
    sample_img_xr = xr.DataArray(
        np.random.rand(3, 40, 40),
        coords=[fovs, np.arange(40), np.arange(40)],
        dims=['fovs', 'x', 'y']
    )

    # basic error checking
    with tempfile.TemporaryDirectory() as temp_dir:
        # bad data_dir path provided
        with pytest.raises(FileNotFoundError):
            data_utils.save_fov_images(fovs, 'bad_data_path', sample_img_xr)

        # invalid fovs provided
        with pytest.raises(ValueError):
            data_utils.save_fov_images(['fov1', 'fov2', 'fov3'], temp_dir, sample_img_xr)

    # test 1: all fovs provided
    with tempfile.TemporaryDirectory() as temp_dir:
        data_utils.save_fov_images(fovs, temp_dir, sample_img_xr)

        for fov in fovs:
            assert os.path.exists(os.path.join(temp_dir, fov + '.tiff'))

    # test 2: fov subset provided
    with tempfile.TemporaryDirectory() as temp_dir:
        data_utils.save_fov_images(fovs_sub, temp_dir, sample_img_xr)

        for fov in fovs_sub:
            assert os.path.exists(os.path.join(temp_dir, fov + '.tiff'))

    # test 3: name suffix provided, along with sub_dir provided
    with tempfile.TemporaryDirectory() as temp_dir:
        data_utils.save_fov_images(fovs_sub, temp_dir, sample_img_xr, sub_dir="sub_directory",
                                   name_suffix='_test')

        for fov in fovs_sub:
            assert os.path.exists(os.path.join(os.path.join(temp_dir, 'sub_directory'),
                                               fov + '_test.tiff'))

    # test 4: name suffix not provided, sub_dir provided
    with tempfile.TemporaryDirectory() as temp_dir:
        data_utils.save_fov_images(fovs_sub, temp_dir, sample_img_xr, sub_dir="sub_directory")
        save_dir = os.path.join(temp_dir, "sub_directory")
        for fov in fovs_sub:
            assert os.path.exists(os.path.join(save_dir, fov + ".tiff"))


def test_generate_deepcell_input():
    for is_mibitiff in [False, True]:
        with tempfile.TemporaryDirectory() as temp_dir:
            fovs = ['fov1', 'fov2', 'fov3']
            chans = ['nuc1', 'nuc2', 'mem1', 'mem2']

            tiff_dir = os.path.join(temp_dir, 'tiff_dir')
            os.mkdir(tiff_dir)

            if is_mibitiff:
                fov_paths, data_xr = test_utils.create_paired_xarray_fovs(
                    tiff_dir, fov_names=fovs, channel_names=chans, mode='mibitiff', dtype='int16'
                )

                # because we're matching files and not directories for mibitiffs
                fovs = [fov + '.tiff' for fov in fovs]
            else:
                fov_paths, data_xr = test_utils.create_paired_xarray_fovs(
                    tiff_dir, fov_names=fovs, channel_names=chans, dtype='int16', sub_dir='TIFs'
                )

            # test 1 nuc, 1 mem (no summing)
            nucs = ['nuc2']
            mems = ['mem2']

            fov1path = os.path.join(temp_dir, 'fov1.tif')
            fov2path = os.path.join(temp_dir, 'fov2.tif')
            fov3path = os.path.join(temp_dir, 'fov3.tif')

            # by setting batch_size=2, we test a batch size with a remainder
            data_utils.generate_deepcell_input(
                data_dir=temp_dir, tiff_dir=tiff_dir, nuc_channels=nucs, mem_channels=mems,
                fovs=fovs, is_mibitiff=is_mibitiff, img_sub_folder='TIFs', batch_size=2
            )

            fov1 = np.moveaxis(io.imread(fov1path), 0, -1)
            fov2 = np.moveaxis(io.imread(fov2path), 0, -1)
            fov3 = np.moveaxis(io.imread(fov3path), 0, -1)

            assert np.array_equal(fov1, data_xr.loc['fov1', :, :, ['nuc2', 'mem2']].values)
            assert np.array_equal(fov2, data_xr.loc['fov2', :, :, ['nuc2', 'mem2']].values)
            assert np.array_equal(fov3, data_xr.loc['fov3', :, :, ['nuc2', 'mem2']].values)

            # test 2 nuc, 2 mem (summing)
            nucs = ['nuc1', 'nuc2']
            mems = ['mem1', 'mem2']

            data_utils.generate_deepcell_input(
                data_dir=temp_dir, tiff_dir=tiff_dir, nuc_channels=nucs, mem_channels=mems,
                fovs=fovs, is_mibitiff=is_mibitiff, img_sub_folder='TIFs', batch_size=2
            )

            nuc_sums = data_xr.loc[:, :, :, nucs].sum(dim='channels').values
            mem_sums = data_xr.loc[:, :, :, mems].sum(dim='channels').values

            fov1 = np.moveaxis(io.imread(fov1path), 0, -1)
            fov2 = np.moveaxis(io.imread(fov2path), 0, -1)
            fov3 = np.moveaxis(io.imread(fov3path), 0, -1)

            assert np.array_equal(fov1[:, :, 0], nuc_sums[0, :, :])
            assert np.array_equal(fov1[:, :, 1], mem_sums[0, :, :])
            assert np.array_equal(fov2[:, :, 0], nuc_sums[1, :, :])
            assert np.array_equal(fov2[:, :, 1], mem_sums[1, :, :])
            assert np.array_equal(fov3[:, :, 0], nuc_sums[2, :, :])
            assert np.array_equal(fov3[:, :, 1], mem_sums[2, :, :])

            # test nuc None
            nucs = None

            data_utils.generate_deepcell_input(
                data_dir=temp_dir, tiff_dir=tiff_dir, nuc_channels=nucs, mem_channels=mems,
                fovs=fovs, is_mibitiff=is_mibitiff, img_sub_folder='TIFs', batch_size=2
            )

            fov1 = np.moveaxis(io.imread(fov1path), 0, -1)
            fov2 = np.moveaxis(io.imread(fov2path), 0, -1)
            fov3 = np.moveaxis(io.imread(fov3path), 0, -1)

            assert np.all(fov1[:, :, 0] == 0)
            assert np.array_equal(fov1[:, :, 1], mem_sums[0, :, :])
            assert np.all(fov2[:, :, 0] == 0)
            assert np.array_equal(fov2[:, :, 1], mem_sums[1, :, :])
            assert np.all(fov3[:, :, 0] == 0)
            assert np.array_equal(fov3[:, :, 1], mem_sums[2, :, :])

            # test mem None
            nucs = ['nuc2']
            mems = None

            data_utils.generate_deepcell_input(
                data_dir=temp_dir, tiff_dir=tiff_dir, nuc_channels=nucs, mem_channels=mems,
                fovs=fovs, is_mibitiff=is_mibitiff, img_sub_folder='TIFs', batch_size=2
            )

            fov1 = np.moveaxis(io.imread(fov1path), 0, -1)
            fov2 = np.moveaxis(io.imread(fov2path), 0, -1)
            fov3 = np.moveaxis(io.imread(fov3path), 0, -1)

            assert np.all(fov1[:, :, 1] == 0)
            assert np.array_equal(fov1[:, :, 0], data_xr.loc['fov1', :, :, 'nuc2'].values)
            assert np.all(fov2[:, :, 1] == 0)
            assert np.array_equal(fov2[:, :, 0], data_xr.loc['fov2', :, :, 'nuc2'].values)
            assert np.all(fov3[:, :, 1] == 0)
            assert np.array_equal(fov3[:, :, 0], data_xr.loc['fov3', :, :, 'nuc2'].values)

            # test nuc None and mem None
            with pytest.raises(ValueError):
                data_utils.generate_deepcell_input(
                    data_xr, temp_dir, None, None, ['fov0'], ['chan0']
                )


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
    data = list(zip(cluster_labels, labels, [fov for _ in range(x * y) for fov in fovs]))
    all_data = pd.DataFrame(data, columns=[
        settings.KMEANS_CLUSTER,
        settings.CELL_LABEL,
        settings.FOV_ID,
    ])
    img_data = np.array([np.arange(1, x * y + 1).reshape((x, y)) for _ in fovs])

    # set random pixels to zero
    idx = np.random.choice(5, 3, replace=False)
    img_data[1][idx] = 0

    np.stack(img_data, axis=0)
    label_maps = xr.DataArray(img_data,
                              coords=[fovs, range(x), range(y)],
                              dims=["fovs", "rows", "cols"])
    res_xr = label_cells_by_cluster([fovs[0]], all_data, label_maps, fov_col=settings.FOV_ID)
    assert res_xr.shape == (1, x, y)

    res_xr = label_cells_by_cluster(fovs, all_data, label_maps, fov_col=settings.FOV_ID)
    assert res_xr.shape == (3, x, y)

    # zero pixels in fov1 should remain zero
    labeled_img = res_xr[res_xr['fovs'] == fovs[1]].values.squeeze()
    assert np.all(labeled_img[idx] == 0)

    # all pixels in fov2 should remain non-zero
    labeled_img = res_xr[res_xr['fovs'] == fovs[2]].values.squeeze()
    assert np.all(labeled_img[idx] > 0)


def test_generate_cell_cluster_mask():
    fovs = ['fov0', 'fov1', 'fov2']
    som_cluster_cols = ['pixel_som_cluster_%d' % i for i in np.arange(5)]
    meta_cluster_cols = ['pixel_meta_cluster_%d' % i for i in np.arange(3)]

    with tempfile.TemporaryDirectory() as temp_dir:
        # bad segmentation path passed
        with pytest.raises(FileNotFoundError):
            data_utils.generate_cell_cluster_mask(
                fovs, temp_dir, 'bad_seg_dir', 'bad_consensus_path'
            )

        # generate sample segmentation masks
        cell_masks = np.random.randint(low=0, high=5, size=(3, 40, 40, 1), dtype="int16")

        for fov in range(cell_masks.shape[0]):
            fov_whole_cell = cell_masks[fov, :, :, 0]
            io.imsave(os.path.join(temp_dir, 'fov%d_feature_0.tif' % fov), fov_whole_cell,
                      check_contrast=False)

        # bad consensus path passed
        with pytest.raises(FileNotFoundError):
            data_utils.generate_cell_cluster_mask(
                fovs, temp_dir, temp_dir, 'bad_consensus_path'
            )

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
            som_data_fov['segmentation_label'] = som_data_fov.index.values + 1
            som_data_fov['cell_som_cluster'] = np.tile(np.arange(1, 6), 4)
            som_data_fov['cell_meta_cluster'] = np.tile(np.arange(1, 3), 10)

            consensus_data_som = pd.concat([consensus_data_som, som_data_fov])

            meta_data_fov = pd.DataFrame(
                np.random.randint(low=0, high=100, size=(20, 3)), columns=meta_cluster_cols
            )

            meta_data_fov['fov'] = fov
            meta_data_fov['segmentation_label'] = meta_data_fov.index.values + 1
            meta_data_fov['cell_som_cluster'] = np.tile(np.arange(1, 6), 4)
            meta_data_fov['cell_meta_cluster'] = np.tile(np.arange(1, 3), 10)

            consensus_data_meta = pd.concat([consensus_data_meta, meta_data_fov])

        # wrote both consensus DataFrames
        feather.write_dataframe(
            consensus_data_som, os.path.join(temp_dir, 'cluster_consensus_som.feather')
        )

        feather.write_dataframe(
            consensus_data_som, os.path.join(temp_dir, 'cluster_consensus_meta.feather')
        )

        # bad cluster column provided
        with pytest.raises(ValueError):
            data_utils.generate_cell_cluster_mask(
                fovs, temp_dir, temp_dir, 'cluster_consensus_som.feather', 'bad_cluster'
            )

        # bad fovs provided
        with pytest.raises(ValueError):
            data_utils.generate_cell_cluster_mask(
                ['fov1', 'fov2', 'fov3'], temp_dir, temp_dir,
                'cluster_consensus_som.feather', 'cell_som_cluster'
            )

        # test on SOM assignments
        cell_masks = data_utils.generate_cell_cluster_mask(
            fovs, temp_dir, temp_dir, 'cluster_consensus_som.feather', 'cell_som_cluster'
        )

        # assert we have 3 fovs and the image size is the same as the mask (40, 40)
        assert cell_masks.shape == (3, 40, 40)

        # assert no value is greater than the highest SOM cluster value (5)
        assert np.all(cell_masks <= 5)

        # test on meta assignments
        cell_masks = data_utils.generate_cell_cluster_mask(
            fovs, temp_dir, temp_dir, 'cluster_consensus_meta.feather', 'cell_meta_cluster'
        )

        # assert we have 3 fovs and the image size is the same as the mask (40, 40)
        assert cell_masks.shape == (3, 40, 40)

        # assert no value is greater than the highest SOM cluster value (2)
        assert np.all(cell_masks <= 2)


def test_generate_pixel_cluster_mask():
    fovs = ['fov0', 'fov1', 'fov2']
    chans = ['chan0', 'chan1', 'chan2', 'chan3']

    with tempfile.TemporaryDirectory() as temp_dir:
        # bad segmentation path passed
        with pytest.raises(FileNotFoundError):
            data_utils.generate_pixel_cluster_mask(
                fovs, temp_dir, 'bad_tiff_dir', 'bad_chan_file', 'bad_consensus_path'
            )

        # bad channel file path passed
        with pytest.raises(FileNotFoundError):
            data_utils.generate_pixel_cluster_mask(
                fovs, temp_dir, temp_dir, 'bad_chan_file', 'bad_consensus_path'
            )

        # generate sample fov folder with one channel value, no sub folder
        channel_data = np.random.randint(low=0, high=5, size=(40, 40), dtype="int16")
        os.mkdir(os.path.join(temp_dir, 'fov0'))
        io.imsave(os.path.join(temp_dir, 'fov0', 'chan0.tif'), channel_data,
                  check_contrast=False)

        # bad consensus path passed
        with pytest.raises(FileNotFoundError):
            data_utils.generate_pixel_cluster_mask(
                fovs, temp_dir, temp_dir, os.path.join('fov0', 'chan0.tif'), 'bad_consensus_path'
            )

        # create a dummy consensus directory
        os.mkdir(os.path.join(temp_dir, 'pixel_mat_consensus'))

        # create dummy data containing SOM and consensus labels for each fov
        for fov in fovs:
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
                fovs, temp_dir, temp_dir, os.path.join('fov0', 'chan0.tif'),
                'pixel_mat_consensus', 'bad_cluster'
            )

        # bad fovs provided
        with pytest.raises(ValueError):
            data_utils.generate_pixel_cluster_mask(
                ['fov1', 'fov2', 'fov3'], temp_dir, temp_dir, os.path.join('fov0', 'chan0.tif'),
                'pixel_mat_consensus', 'pixel_som_cluster'
            )

        # test on SOM assignments
        pixel_masks = data_utils.generate_pixel_cluster_mask(
            fovs, temp_dir, temp_dir, os.path.join('fov0', 'chan0.tif'),
            'pixel_mat_consensus', 'pixel_som_cluster'
        )

        # assert we have 3 fovs and the image size is the same as the mask (40, 40)
        assert pixel_masks.shape == (3, 40, 40)

        # assert no value is greater than the highest SOM cluster value (10)
        assert np.all(pixel_masks <= 10)

        # test on meta assignments
        pixel_masks = data_utils.generate_pixel_cluster_mask(
            fovs, temp_dir, temp_dir, os.path.join('fov0', 'chan0.tif'),
            'pixel_mat_consensus', 'pixel_meta_cluster'
        )

        # assert we have 3 fovs and the image size is the same as the mask (40, 40)
        assert pixel_masks.shape == (3, 40, 40)

        # assert no value is greater than the highest meta cluster value (5)
        assert np.all(pixel_masks <= 5)
