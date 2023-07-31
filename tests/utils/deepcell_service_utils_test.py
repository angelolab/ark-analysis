import itertools
import os
import pathlib
import tempfile
import time
from zipfile import ZipFile
from pathlib import Path

import numpy as np
import pytest
import tifffile
from alpineer import image_utils, test_utils, io_utils
from pytest_mock import MockerFixture
from skimage import io
from unittest.mock import call, patch

from ark.utils import deepcell_service_utils
from ark.utils.deepcell_service_utils import (_convert_deepcell_seg_masks, create_deepcell_output,
                                              zip_input_files, extract_deepcell_response)


def mocked_zip_input(input_dir, fovs, batch_num):
    fov_data = np.ones(shape=(10, 10), dtype="float32")
    for fov in fovs:
        image_utils.save_image(os.path.join(input_dir, f'{fov}.tiff'), fov_data)

    zip_input_files(input_dir, fovs, batch_num)


def mocked_bad_run_deepcell(in_zip_path, output_dir, host, job_type, scale, timeout):
    return mocked_run_deepcell(
        in_zip_path, output_dir, host, job_type, scale, timeout, missing=True)


def mocked_run_deepcell(in_zip_path, output_dir, host, job_type, scale, timeout, missing=False):
    fov_data = np.ones(shape=(10, 10), dtype="float32")
    with ZipFile(in_zip_path, 'r') as zipObj:
        fovs = io_utils.remove_file_extensions(zipObj.namelist())

    if missing:
        fov_seg_pairs = list(itertools.product(fovs, ['feature_0']))
    else:
        fov_seg_pairs = list(itertools.product(fovs, ['feature_0', 'feature_1']))

    # temp write output files for zip
    for fov, seg_type in fov_seg_pairs:
        image_utils.save_image(os.path.join(output_dir, f'{fov}_{seg_type}.tif'), fov_data)

    batch_num = int(in_zip_path.split('.')[0].split('_')[-1])
    zip_path = os.path.join(output_dir, f'deepcell_response_fovs_batch_{batch_num}.zip')
    # write deepcell output into zip file
    with ZipFile(zip_path, 'w') as zipObj:
        for fov, seg_type in fov_seg_pairs:
            filename = os.path.join(output_dir, f'{fov}_{seg_type}.tif')
            zipObj.write(filename, os.path.basename(filename))
            os.remove(filename)

    return 0


def test_zip_input_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        fov_data = np.ones(shape=(10, 10), dtype="float32")
        image_utils.save_image(os.path.join(temp_dir, 'fov1.tiff'), fov_data)
        image_utils.save_image(os.path.join(temp_dir, 'fov2.tiff'), fov_data)

        # test successful zipping
        zip_path = zip_input_files(temp_dir, fov_group=["fov1", "fov2"], batch_num=1)
        create_time = Path(zip_path).stat().st_ctime

        # check zip contents
        with ZipFile(os.path.join(temp_dir, 'fovs_batch_1.zip'), 'r') as zip_batch1:
            assert zip_batch1.namelist() == ['fov1.tiff', 'fov2.tiff']

        # test previously zipped batches are not re-zipped
        time.sleep(3)
        zip_path = zip_input_files(temp_dir, fov_group=["fov1", "fov2"], batch_num=1)
        modify_time = Path(zip_path).stat().st_mtime

        # check zip file was not overwritten
        assert np.isclose(modify_time, create_time)


def test_extract_deepcell_response():
    with tempfile.TemporaryDirectory() as temp_dir:
        mocked_zip_input(temp_dir, ["fov1", "fov2", "fov3"], 1)
        mocked_run_deepcell(os.path.join(temp_dir, "fovs_batch_1.zip"), temp_dir,
                            host='https://deepcell.org', job_type='mesmer', scale=1.0,
                            timeout=3600)

        # test successful extraction
        extract_deepcell_response(temp_dir, ["fov1", "fov2", "fov3"], 1, wc_suffix="_whole_cell",
                                  nuc_suffix="_nuclear")

        assert os.path.exists(os.path.join(temp_dir, 'fov1_whole_cell.tiff'))
        assert os.path.exists(os.path.join(temp_dir, 'fov1_nuclear.tiff'))
        assert os.path.exists(os.path.join(temp_dir, 'fov2_whole_cell.tiff'))
        assert os.path.exists(os.path.join(temp_dir, 'fov2_nuclear.tiff'))
        assert os.path.exists(os.path.join(temp_dir, 'fov3_whole_cell.tiff'))
        assert os.path.exists(os.path.join(temp_dir, 'fov3_nuclear.tiff'))

    with tempfile.TemporaryDirectory() as temp_dir:
        mocked_zip_input(temp_dir, ["fov1", "fov2", "fov3"], 1)
        mocked_bad_run_deepcell(os.path.join(temp_dir, "fovs_batch_1.zip"), temp_dir,
                                host='https://deepcell.org', job_type='mesmer', scale=1.0,
                                timeout=3600)

        # DeepCell nuclear output .tif file does not exist
        with pytest.warns(UserWarning):
            extract_deepcell_response(temp_dir, ["fov1", "fov2", "fov3"], 1,
                                      wc_suffix="_whole_cell", nuc_suffix="_nuclear")


def test_create_deepcell_output(mocker: MockerFixture):
    with tempfile.TemporaryDirectory() as temp_dir:
        mocker.patch('ark.utils.deepcell_service_utils.run_deepcell_direct', mocked_run_deepcell)

        input_dir = os.path.join(temp_dir, 'input_dir')
        os.makedirs(input_dir)

        fov_data = np.ones(shape=(10, 10), dtype="float32")
        image_utils.save_image(os.path.join(input_dir, 'fov1.tiff'), fov_data)
        image_utils.save_image(os.path.join(input_dir, 'fov2.tiff'), fov_data)
        image_utils.save_image(os.path.join(input_dir, 'fov3.tiff'), fov_data)

        with tempfile.TemporaryDirectory() as output_dir:

            with pytest.raises(ValueError):
                # fail if non-existent fovs are specified
                create_deepcell_output(deepcell_input_dir=input_dir,
                                       deepcell_output_dir=output_dir, fovs=['fov1', 'fov1000'])

            # test with specified fov list
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   fovs=['fov1', 'fov2'])

            with pytest.raises(ValueError):
                # fail if scale argument can not be converted to float
                create_deepcell_output(deepcell_input_dir=input_dir,
                                       deepcell_output_dir=output_dir, fovs=['fov1', 'fov2'],
                                       scale='test')

            # make sure DeepCell (.zip) output exists
            assert os.path.exists(os.path.join(output_dir, 'deepcell_response_fovs_batch_1.zip'))

            # DeepCell output .zip file should be extracted
            assert os.path.exists(os.path.join(output_dir, 'fov1_whole_cell.tiff'))
            assert os.path.exists(os.path.join(output_dir, 'fov1_nuclear.tiff'))
            assert os.path.exists(os.path.join(output_dir, 'fov2_whole_cell.tiff'))
            assert os.path.exists(os.path.join(output_dir, 'fov2_nuclear.tiff'))
            # check fov3 not processed
            assert not os.path.exists(os.path.join(output_dir, 'fov3_whole_cell.tiff'))

            # test for 2d shape
            whole_cell_arr = io.imread(os.path.join(output_dir, 'fov1_whole_cell.tiff'))
            nuclear_arr = io.imread(os.path.join(output_dir, 'fov1_nuclear.tiff'))
            assert len(whole_cell_arr.shape) == len(nuclear_arr.shape) == 2

            for batch_zip in io_utils.list_files(input_dir, substrs=".zip"):
                os.remove(os.path.join(input_dir, batch_zip))

        with tempfile.TemporaryDirectory() as output_dir:

            # test with mixed fov/file list
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   fovs=['fov1', 'fov2.tiff', 'fov3.tiff'])

            # make sure DeepCell (.zip) output exists
            assert os.path.exists(os.path.join(output_dir, 'deepcell_response_fovs_batch_1.zip'))
            assert os.path.exists(os.path.join(output_dir, 'fov1_whole_cell.tiff'))
            assert os.path.exists(os.path.join(output_dir, 'fov2_whole_cell.tiff'))
            assert os.path.exists(os.path.join(output_dir, 'fov3_whole_cell.tiff'))

            for batch_zip in io_utils.list_files(input_dir, substrs=".zip"):
                os.remove(os.path.join(input_dir, batch_zip))

        with tempfile.TemporaryDirectory() as output_dir:
            # if fovs is None, all .tiff files in input dir should be taken
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir)

            # make sure DeepCell (.zip) output exists
            assert os.path.exists(os.path.join(output_dir, 'deepcell_response_fovs_batch_1.zip'))
            with ZipFile(os.path.join(output_dir, 'deepcell_response_fovs_batch_1.zip'), 'r') \
                    as zip_batch1:
                assert sorted(zip_batch1.namelist()) == \
                       ['fov1_feature_0.tif', 'fov1_feature_1.tif', 'fov2_feature_0.tif',
                        'fov2_feature_1.tif', 'fov3_feature_0.tif', 'fov3_feature_1.tif']

            for batch_zip in io_utils.list_files(input_dir, substrs=".zip"):
                os.remove(os.path.join(input_dir, batch_zip))

        with tempfile.TemporaryDirectory() as output_dir:
            # add additional fov for auto-batch testing
            pathlib.Path(os.path.join(input_dir, 'fov4.tiff')).touch()

            create_deepcell_output(deepcell_input_dir=input_dir,
                                   deepcell_output_dir=output_dir,
                                   fovs=['fov1', 'fov2', 'fov3', 'fov4'], zip_size=3)

            # check that there are two zip files with sizes 3, 1 respectively
            assert os.path.exists(os.path.join(input_dir, 'fovs_batch_1.zip'))
            assert os.path.exists(os.path.join(input_dir, 'fovs_batch_2.zip'))

            with ZipFile(os.path.join(input_dir, 'fovs_batch_1.zip'), 'r') as zip_batch1:
                assert zip_batch1.namelist() == ['fov1.tiff', 'fov2.tiff', 'fov3.tiff']
            with ZipFile(os.path.join(input_dir, 'fovs_batch_2.zip'), 'r') as zip_batch2:
                assert zip_batch2.namelist() == ['fov4.tiff']
            # check output for extra fov batch
            assert os.path.exists(os.path.join(output_dir, "deepcell_response_fovs_batch_2.zip"))

            for batch_zip in io_utils.list_files(input_dir, substrs=".zip"):
                os.remove(os.path.join(input_dir, batch_zip))

        with tempfile.TemporaryDirectory() as output_dir:
            # ValueError should be raised if .tiff file does not exist for some fov in fovs
            with pytest.raises(ValueError):
                create_deepcell_output(deepcell_input_dir=input_dir,
                                       deepcell_output_dir=output_dir, fovs=['fov1', 'fov5'])


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

            fov1path = os.path.join(temp_dir, 'fov1.tiff')
            fov2path = os.path.join(temp_dir, 'fov2.tiff')
            fov3path = os.path.join(temp_dir, 'fov3.tiff')

            deepcell_service_utils.generate_deepcell_input(
                data_dir=temp_dir, tiff_dir=tiff_dir, nuc_channels=nucs, mem_channels=mems,
                fovs=fovs, is_mibitiff=is_mibitiff, img_sub_folder='TIFs'
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

            deepcell_service_utils.generate_deepcell_input(
                data_dir=temp_dir, tiff_dir=tiff_dir, nuc_channels=nucs, mem_channels=mems,
                fovs=fovs, is_mibitiff=is_mibitiff, img_sub_folder='TIFs'
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

            deepcell_service_utils.generate_deepcell_input(
                data_dir=temp_dir, tiff_dir=tiff_dir, nuc_channels=nucs, mem_channels=mems,
                fovs=fovs, is_mibitiff=is_mibitiff, img_sub_folder='TIFs'
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

            deepcell_service_utils.generate_deepcell_input(
                data_dir=temp_dir, tiff_dir=tiff_dir, nuc_channels=nucs, mem_channels=mems,
                fovs=fovs, is_mibitiff=is_mibitiff, img_sub_folder='TIFs'
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
                deepcell_service_utils.generate_deepcell_input(
                    data_xr, temp_dir, None, None, ['fov0'], ['chan0']
                )


def test_convert_deepcell_seg_masks():
    with tempfile.TemporaryDirectory() as temp_dir:

        # Initialize a new generator - set seed for reproducibility
        rng = np.random.default_rng(12345)

        # Create a test mask with integers and cast them to floating point.
        true_test_mask = np.array([[4, 1, 5, 2, 1], [5, 4, 4, 6, 2], [5, 2, 3, 4, 1],
                                   [1, 1, 4, 4, 6], [4, 1, 6, 6, 5]], dtype="int32")

        test_mask = rng.integers(low=0, high=7, size=(5, 5)).astype("float32")

        tifffile.imwrite(f"{temp_dir}/test_mask.tiff", data=test_mask)

        with open(f"{temp_dir}/test_mask.tiff", 'r+b') as test_mask_bytes:
            processed_mask = _convert_deepcell_seg_masks(test_mask_bytes.read())

            assert np.issubdtype(processed_mask.dtype, np.integer)
            np.testing.assert_equal(true_test_mask, processed_mask)
