import os
import tempfile
from pathlib import Path
from unittest.mock import patch


from ark.segmentation import fiber_segmentation
from ark.utils.test_utils import _write_tifs


#@patch("ark.segmentation.fiber_segmentation.plt.imshow")
def test_plot_fiber_segmentation_steps(mock_plt):
    img_dir = os.path.join(Path(__file__).parent.parent.parent,
                            "data", "example_dataset", "fiber_segmentation")

    fiber_segmentation.plot_fiber_segmentation_steps(img_dir, 'fov1', 'Collagen1')


def test_batch_segment_fibers():
    with tempfile.TemporaryDirectory() as temp_dir:
        img_dir = os.path.join(Path(__file__).parent.parent.parent,
                               "data", "example_dataset", "fiber_segmentation")

        fiber_object_table, fiber_label_images = fiber_segmentation.batch_segment_fibers(
            img_dir, 'Collagen1', temp_dir, batch_size=1, )

        assert os.path.exists(os.path.join(temp_dir, 'fiber_object_table.csv'))
