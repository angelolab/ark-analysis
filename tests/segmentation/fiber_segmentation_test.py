import os
import shutil
import tempfile

import pytest
from alpineer import io_utils

import ark.settings as settings
from ark.segmentation import fiber_segmentation
from ark.utils import example_dataset


def test_plot_fiber_segmentation_steps():

    with tempfile.TemporaryDirectory() as temp_dir:
        # download example data, keep only 3 fovs for testing
        example_dataset.get_example_dataset(dataset="segment_image_data", save_dir=temp_dir)
        img_dir = os.path.join(temp_dir, 'image_data')
        for fov in ['fov3', 'fov4', 'fov5', 'fov6', 'fov7', 'fov8', 'fov9', 'fov10']:
            shutil.rmtree(os.path.join(temp_dir, 'image_data', fov))

        # bad directory should raise an errors
        with pytest.raises(FileNotFoundError):
            _, _ = fiber_segmentation.plot_fiber_segmentation_steps('bad_dir', 'fov1', 'Collagen1')

        # bad channel should raise an errors
        with pytest.raises(ValueError):
            _, _ = fiber_segmentation.plot_fiber_segmentation_steps(img_dir, 'fov1', 'bad_channel')

        # bad subdirectory should raise an errors
        with pytest.raises(FileNotFoundError):
            _, _ = fiber_segmentation.plot_fiber_segmentation_steps(
                img_dir, 'fov1', 'Collagen1', img_sub_folder='bad_subdir')

        # test success
        fiber_segmentation.plot_fiber_segmentation_steps(img_dir, 'fov1', 'Collagen1')


def test_run_fiber_segmentation():
    with tempfile.TemporaryDirectory() as temp_dir:

        # download example data, keep only 3 fovs for testing
        example_dataset.get_example_dataset(dataset="segment_image_data", save_dir=temp_dir)
        img_dir = os.path.join(temp_dir, 'image_data')
        for fov in ['fov3', 'fov4', 'fov5', 'fov6', 'fov7', 'fov8', 'fov9', 'fov10']:
            shutil.rmtree(os.path.join(temp_dir, 'image_data', fov))
        out_dir = os.path.join(temp_dir, 'fiber_segmentation')
        os.makedirs(out_dir)

        # bad directories should raise an error
        with pytest.raises(FileNotFoundError):
            _ = fiber_segmentation.run_fiber_segmentation('bad_path', 'Collagen1', out_dir)

        with pytest.raises(FileNotFoundError):
            _ = fiber_segmentation.run_fiber_segmentation(img_dir, 'Collagen1', 'bad_path')

        # bad subdirectory should raise an errors
        with pytest.raises(FileNotFoundError):
            _ = fiber_segmentation.run_fiber_segmentation(img_dir, 'Collagen1', out_dir,
                                                          img_sub_folder='bad_folder')

        # bad channel should raise an errors
        with pytest.raises(ValueError):
            _ = fiber_segmentation.run_fiber_segmentation(img_dir, 'bad_channel', out_dir)

        # test success
        fiber_object_table = fiber_segmentation.run_fiber_segmentation(
            img_dir, 'Collagen1', out_dir)

        # check all fovs are processed
        assert fiber_object_table[settings.FOV_ID].unique().sort() == \
               io_utils.list_folders(img_dir).sort()

        # check output files
        for fov in io_utils.list_files(img_dir):
            assert os.path.exists(os.path.join(out_dir, f'{fov}_fiber_labels.tiff'))
        assert os.path.exists(os.path.join(out_dir, 'fiber_object_table.csv'))

        # test success with debugging
        fiber_object_table = fiber_segmentation.run_fiber_segmentation(
            img_dir, 'Collagen1', out_dir, debug=True)

        # check debug output files
        intermediate_imgs = ['fov1_thresholded.tiff', 'fov1_ridges_thresholded.tiff',
                             'fov1_meijering_filter.tiff', 'fov1_contrast_adjusted.tiff']
        for img in intermediate_imgs:
            img_path = os.path.join(out_dir, '_debug', img)
            assert os.path.exists(img_path)
