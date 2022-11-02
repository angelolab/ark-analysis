import os
import pathlib
import tempfile
from zipfile import ZipFile

import numpy as np
import pytest
import tifffile
from pytest_mock import MockerFixture
from skimage import io

from ark.utils.deepcell_service_utils import (_convert_deepcell_seg_masks,
                                              create_deepcell_output)


def mocked_run_deepcell(in_zip_path, output_dir, host, job_type, scale, timeout):

    fov_data = np.ones(shape=(10, 10), dtype="float32")
    io.imsave(os.path.join(output_dir, 'fov1_feature_0.tif'),
              fov_data, plugin="tifffile", check_contrast=False)
    io.imsave(os.path.join(output_dir, 'fov2_feature_0.tif'),
              fov_data, plugin="tifffile", check_contrast=False)
    io.imsave(os.path.join(output_dir, 'fov3_feature_0.tif'),
              fov_data, plugin="tifffile", check_contrast=False)

    batch_num = int(in_zip_path.split('.')[0].split('_')[-1])
    if batch_num < 2:
        zip_path = os.path.join(output_dir, 'example_output.zip')
    else:
        zip_path = os.path.join(output_dir, f'example_output_{batch_num}.zip')
    with ZipFile(zip_path, 'w') as zipObj:
        if batch_num > 1:
            return
        for i in range(1, 4):
            filename = os.path.join(output_dir, f'fov{i}_feature_0.tif')
            zipObj.write(filename, os.path.basename(filename))
            os.remove(filename)

    return 0


def test_create_deepcell_output(mocker: MockerFixture):
    with tempfile.TemporaryDirectory() as temp_dir:
        mocker.patch('ark.utils.deepcell_service_utils.run_deepcell_direct', mocked_run_deepcell)

        input_dir = os.path.join(temp_dir, 'input_dir')
        os.makedirs(input_dir)

        fov_data = np.ones(shape=(10, 10), dtype="float32")
        io.imsave(os.path.join(input_dir, 'fov1.tiff'),
                  fov_data, plugin="tifffile", check_contrast=False)
        io.imsave(os.path.join(input_dir, 'fov2.tiff'),
                  fov_data, plugin="tifffile", check_contrast=False)
        io.imsave(os.path.join(input_dir, 'fov3.tiff'),
                  fov_data, plugin="tifffile", check_contrast=False)

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
            assert os.path.exists(os.path.join(output_dir, 'example_output.zip'))

            # DeepCell output .zip file should be extracted
            assert os.path.exists(os.path.join(output_dir, 'fov1_feature_0.tiff'))
            assert os.path.exists(os.path.join(output_dir, 'fov2_feature_0.tiff'))

        with tempfile.TemporaryDirectory() as output_dir:

            # test parallel
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   fovs=['fov1', 'fov2'], zip_size=1, parallel=True)

            # make sure DeepCell (.zip's) output exists
            assert os.path.exists(os.path.join(output_dir, 'example_output.zip'))
            assert os.path.exists(os.path.join(output_dir, 'example_output_2.zip'))

        with tempfile.TemporaryDirectory() as output_dir:

            # test with mixed fov/file list
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir,
                                   fovs=['fov1', 'fov2.tiff', 'fov3.tiff'])

            # make sure DeepCell (.zip) output exists
            assert os.path.exists(os.path.join(output_dir, 'example_output.zip'))

            # DeepCell output .zip file should be extracted
            assert os.path.exists(os.path.join(output_dir, 'fov1_feature_0.tiff'))
            assert os.path.exists(os.path.join(output_dir, 'fov2_feature_0.tiff'))
            assert os.path.exists(os.path.join(output_dir, 'fov3_feature_0.tiff'))

        with tempfile.TemporaryDirectory() as output_dir:

            # if fovs is None, all .tiff files in input dir should be taken
            create_deepcell_output(deepcell_input_dir=input_dir, deepcell_output_dir=output_dir)

            # make sure DeepCell (.zip) output exists
            assert os.path.exists(os.path.join(output_dir, 'example_output.zip'))

            assert os.path.exists(os.path.join(output_dir, 'fov1_feature_0.tiff'))
            assert os.path.exists(os.path.join(output_dir, 'fov2_feature_0.tiff'))
            assert os.path.exists(os.path.join(output_dir, 'fov3_feature_0.tiff'))

            pathlib.Path(os.path.join(input_dir, 'fovs.zip')).touch()

            # Warning should be displayed if fovs.zip file exists (will be overwritten)
            with pytest.warns(UserWarning):
                create_deepcell_output(deepcell_input_dir=input_dir,
                                       deepcell_output_dir=output_dir, fovs=['fov1'])

            # DeepCell output .tif file does not exist for some fov
            with pytest.warns(UserWarning):
                create_deepcell_output(deepcell_input_dir=input_dir,
                                       deepcell_output_dir=output_dir, suffix='_other_suffix',
                                       fovs=['fov1'])

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

            # ValueError should be raised if .tiff file does not exists for some fov in fovs
            with pytest.raises(ValueError):
                create_deepcell_output(deepcell_input_dir=input_dir,
                                       deepcell_output_dir=output_dir, fovs=['fov1', 'fov5'])


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
