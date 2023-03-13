import os
import shutil
import tempfile
from random import randint
from shutil import rmtree

import feather
import numpy as np
import pandas as pd
import pytest
import skimage.io as io
import xarray as xr
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


def test_relabel_segmentation():
    x = y = 5
    img_arr = np.arange(1, x * y + 1).reshape((x, y))
    d = {i: i + 1 for i in range(1, x * y + 1)}
    res = data_utils.relabel_segmentation(img_arr, d)

    assert np.array_equal(img_arr + 1, res)

    # some cells are not mapped to any cluster-label
    d = {i: i + 1 for i in range(1, x * y - 5)}
    res = data_utils.relabel_segmentation(img_arr, d)
    # these cells should all get a default label
    img_arr[img_arr >= x * y - 5] = x * y - 5

    assert np.array_equal(img_arr + 1, res)

    # test case for multiple pixels with the same label
    data = np.array([[1, 2], [3, 4]])
    data = np.repeat(data, 2)  # ([1, 1, 2, 2, 3, 3, 4, 4])
    img_arr = data.reshape((4, 2))
    d = {i: 10 * i for i in range(5)}
    res = data_utils.relabel_segmentation(img_arr, d)

    assert np.array_equal(img_arr * 10, res)


@parametrize('zero_pixels', [False, True])
def test_label_cells_by_cluster(zero_pixels):
    # define a sample FOV name
    fov = 'fov0'

    # define the dimensions
    x = y = 5

    # define a sample all_data
    cluster_labels = np.random.randint(1, 5, x * y)
    labels = [i % (x * y) for i in range(x * y)]
    data = list(zip(cluster_labels, labels, [fov for _ in range(x * y)]))
    all_data = pd.DataFrame(data, columns=[
        settings.KMEANS_CLUSTER,
        settings.CELL_LABEL,
        settings.FOV_ID,
    ])

    img_data = np.array(np.arange(1, x * y + 1).reshape((x, y)))
    if zero_pixels:
        img_data = 0

    # define a label map for the FOV
    label_map = xr.DataArray(
        img_data, coords=[range(x), range(y)], dims=['rows', 'cols']
    )

    # relabel the cells
    res_data = data_utils.label_cells_by_cluster(fov, all_data, label_map, fov_col=settings.FOV_ID)

    # assert the shape is the same as the original label_map
    assert res_data.shape == (x, y)

    # assert the pixels are zero or non-zero per the test
    test_mask = res_data == 0 if zero_pixels else res_data > 0
    assert np.all(test_mask)


def test_generate_cell_cluster_mask():
    fov = 'fov0'
    som_cluster_cols = ['pixel_som_cluster_%d' % i for i in np.arange(5)]
    meta_cluster_cols = ['pixel_meta_cluster_%d' % i for i in np.arange(3)]

    with tempfile.TemporaryDirectory() as temp_dir:
        # bad segmentation path passed
        with pytest.raises(FileNotFoundError):
            data_utils.generate_cell_cluster_mask(
                fov, temp_dir, 'bad_seg_dir', pd.DataFrame()
            )

        # generate a sample segmentation mask
        cell_mask = np.random.randint(low=0, high=5, size=(40, 40), dtype="int16")
        image_utils.save_image(os.path.join(temp_dir, '%s_whole_cell.tiff' % fov), cell_mask)

        # create a sample cell consensus file based on SOM cluster assignments
        consensus_data_som = pd.DataFrame(
            np.random.randint(low=0, high=100, size=(20, 5)), columns=som_cluster_cols
        )

        consensus_data_som['fov'] = fov
        consensus_data_som['segmentation_label'] = consensus_data_som.index.values + 1
        consensus_data_som['cell_som_cluster'] = np.tile(np.arange(1, 6), 4)
        consensus_data_som['cell_meta_cluster'] = np.tile(np.arange(1, 3), 10)

        # create a sample cell consensus file based on meta cluster assignments
        consensus_data_meta = pd.DataFrame(
            np.random.randint(low=0, high=100, size=(20, 3)), columns=meta_cluster_cols
        )

        consensus_data_meta['fov'] = fov
        consensus_data_meta['segmentation_label'] = consensus_data_meta.index.values + 1
        consensus_data_meta['cell_som_cluster'] = np.tile(np.arange(1, 6), 4)
        consensus_data_meta['cell_meta_cluster'] = np.tile(np.arange(1, 3), 10)

        # bad cluster column provided
        with pytest.raises(ValueError):
            data_utils.generate_cell_cluster_mask(
                fov, temp_dir, temp_dir, consensus_data_som, 'bad_cluster'
            )

        # bad fov provided
        with pytest.raises(ValueError):
            data_utils.generate_cell_cluster_mask(
                'fov1', temp_dir, temp_dir,
                consensus_data_som, 'cell_som_cluster'
            )

        # test on SOM assignments
        cell_masks = data_utils.generate_cell_cluster_mask(
            fov, temp_dir, temp_dir, consensus_data_som, 'cell_som_cluster'
        )

        # assert the image size is the same as the mask (40, 40)
        assert cell_masks.shape == (40, 40)

        # assert no value is greater than the highest SOM cluster value (5)
        assert np.all(cell_masks <= 5)

        # test on meta assignments
        cell_masks = data_utils.generate_cell_cluster_mask(
            fov, temp_dir, temp_dir, consensus_data_meta, 'cell_meta_cluster'
        )

        # assert the image size is the same as the mask (40, 40)
        assert cell_masks.shape == (40, 40)

        # assert no value is greater than the highest SOM cluster value (2)
        assert np.all(cell_masks <= 2)


@parametrize('sub_dir', [None, 'sub_dir'])
@parametrize('name_suffix', ['', 'sample_suffix'])
def test_generate_and_save_cell_cluster_masks(sub_dir, name_suffix):
    fov_count = 7
    fovs = [f"fov{i}" for i in range(fov_count)]
    som_cluster_cols = ['pixel_som_cluster_%d' % i for i in np.arange(5)]
    meta_cluster_cols = ['pixel_meta_cluster_%d' % i for i in np.arange(3)]
    fov_size_split = 4

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a save directory
        os.mkdir(os.path.join(temp_dir, 'cell_masks'))

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
            image_utils.save_image(os.path.join(temp_dir, 'fov%d_whole_cell.tiff' % fov),
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

        # test various batch_sizes, no sub_dir, name_suffix = ''.
        data_utils.generate_and_save_cell_cluster_masks(
            fovs=fovs,
            base_dir=temp_dir,
            save_dir=os.path.join(temp_dir, 'cell_masks'),
            seg_dir=temp_dir,
            cell_data=consensus_data_som,
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
            cell_mask = io.imread(os.path.join(temp_dir, 'cell_masks', sub_dir, fov_name))
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
            neighborhood_data=sample_neighborhood_data,
            seg_dir=os.path.join(temp_dir, 'seg_dir'),
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
            assert np.all(np.isin(neighborhood_mask, np.array([10 * i for i in np.arange(6)])))


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


@pytest.mark.parametrize('segmentation, clustering, subdir',
                         [(False, False, 'TIFs'), (True, False, ''), (False, 'cell', ''),
                          (False, 'pixel', '')])
@pytest.mark.parametrize('fovs', [['R1C1', 'R2C2', 'R3C1'],
                         ['run_1_R1C1', 'run_1_R2C2', 'run_2_R3C1']])
def test_stitch_images_by_shape(segmentation, clustering, subdir, fovs):

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
            os.makedirs(os.path.join(temp_dir, 'R1C1'))
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
            chans = [f'chan{i}' for i in range(5)]
            # check that ignores toffy stitching in fov level dir
            fovs.append('stitched_images')

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            data_dir, fovs, chans,
            img_shape=(10, 10), fills=True, sub_dir=subdir, dtype=np.float32,
            single_dir=any([segmentation, clustering])
        )

        # bad channel name should raise an error
        with pytest.raises(ValueError, match="Not all values given in list"):
            data_utils.stitch_images_by_shape(data_dir, stitched_dir, channels='bad_channel',
                                              img_sub_folder=subdir, segmentation=segmentation,
                                              clustering=clustering)

        # test successful stitching
        data_utils.stitch_images_by_shape(data_dir, stitched_dir, img_sub_folder=subdir,
                                          segmentation=segmentation, clustering=clustering)
        assert sorted(io_utils.list_files(stitched_dir)) == \
            [chan + '_stitched.tiff' for chan in chans]

        # stitched image is 3 x 2 fovs with max_img_size = 10
        stitched_data = load_utils.load_imgs_from_dir(stitched_dir,
                                                      files=[chans[0] + '_stitched.tiff'])
        assert stitched_data.shape == (1, 30, 20, 1)
        shutil.rmtree(stitched_dir)

        # test successful stitching for select channels
        random_channel = chans[randint(0, len(chans)-1)]
        data_utils.stitch_images_by_shape(data_dir, stitched_dir, img_sub_folder=subdir,
                                          channels=[random_channel], segmentation=segmentation,
                                          clustering=clustering)
        assert sorted(io_utils.list_files(stitched_dir)) == [random_channel + '_stitched.tiff']

        # remove stitched_images from fov list
        if not segmentation and not clustering:
            fovs.pop()
