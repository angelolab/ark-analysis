import os
import tempfile
import pytest
from pathlib import Path

import ark.settings as settings
from ark.utils import io_utils
from ark.segmentation import fiber_segmentation


def test_plot_fiber_segmentation_steps():
    img_dir = os.path.join(Path(__file__).parent.parent.parent,
                           "data", "example_dataset", "fiber_segmentation")

    fiber_segmentation.plot_fiber_segmentation_steps(img_dir, 'fov1', 'Collagen1')

    # bad directory should raise an errors
    with pytest.raises(ValueError):
        _, _ = fiber_segmentation.plot_fiber_segmentation_steps('bad_dir', 'fov1', 'Collagen1')

    # bad channel should raise an errors
    with pytest.raises(ValueError):
        _, _ = fiber_segmentation.plot_fiber_segmentation_steps(img_dir, 'fov1', 'bad_channel')

    # bad subdirectory should raise an errors
    with pytest.raises(ValueError):
        _, _ = fiber_segmentation.plot_fiber_segmentation_steps(
            img_dir, 'fov1', 'Collagen1', img_sub_folder='bad_subdir')


def test_run_fiber_segmentation():
    with tempfile.TemporaryDirectory() as temp_dir:

        img_dir = os.path.join(Path(__file__).parent.parent.parent,
                               "data", "example_dataset", "fiber_segmentation")

        # bad directories should raise an error
        with pytest.raises(ValueError):
            _, _ = fiber_segmentation.run_fiber_segmentation('bad_path', 'Collagen1', temp_dir)

        with pytest.raises(ValueError):
            _, _ = fiber_segmentation.run_fiber_segmentation(img_dir, 'Collagen1', 'bad_path')

        # bad subdirectory should raise an errors
        with pytest.raises(ValueError):
            _, _ = fiber_segmentation.plot_fiber_segmentation_steps(
                img_dir, 'fov1', 'Collagen1', img_sub_folder='bad_subdir')

        # bad channel should raise an errors
        with pytest.raises(ValueError):
            _, _ = fiber_segmentation.run_fiber_segmentation(img_dir, 'bad_channel', temp_dir)

        # test success
        fiber_object_table, fiber_label_images = fiber_segmentation.run_fiber_segmentation(
            img_dir, 'Collagen1', temp_dir)

        # check all fovs are processed
        assert fiber_object_table[settings.FOV_ID].unique().sort() == \
               io_utils.list_folders(img_dir).sort()

        # check output files
        for fov in io_utils.list_files(img_dir):
            assert os.path.exists(os.path.join(temp_dir, f'{fov}_fiber_labels.tiff'))
        assert os.path.exists(os.path.join(temp_dir, 'fiber_object_table.csv'))

        # test success with debugging
        fiber_object_table, fiber_label_images = fiber_segmentation.run_fiber_segmentation(
            img_dir, 'Collagen1', temp_dir, debug=True)

        # check debug output files
        intermediate_imgs = ['fov1_thresholded.tiff', 'fov1_ridges_thresholded.tiff',
                             'fov1_meijering_filter.tiff', 'fov1_contrast_adjusted.tiff']
        for img in intermediate_imgs:
            img_path = os.path.join(temp_dir, '_debug', img)
            assert os.path.exists(img_path)
