import tempfile
import os

from ark.analysis import dimensionality_reduction
from ark.utils import test_utils


def test_dimensionality_reduction():
    random_cell_data = test_utils.make_segmented_csv(300)
    test_cols = test_utils.TEST_MARKERS

    test_algorithms = ['PCA', 'tSNE', 'UMAP']
    with tempfile.TemporaryDirectory() as temp_dir:
        for alg in test_algorithms:
            dimensionality_reduction.visualize_dimensionality_reduction(random_cell_data,
                                                                        test_cols,
                                                                        "cell_type",
                                                                        algorithm=alg)
            assert not os.path.exists(os.path.join(temp_dir, alg + 'Visualization.png'))

        for alg in test_algorithms:
            dimensionality_reduction.visualize_dimensionality_reduction(random_cell_data,
                                                                        test_cols,
                                                                        "cell_type",
                                                                        algorithm=alg,
                                                                        save_dir=temp_dir)
            assert os.path.exists(os.path.join(temp_dir, alg + 'Visualization.png'))
