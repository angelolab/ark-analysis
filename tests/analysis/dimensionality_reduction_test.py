import os
import tempfile

import pytest

import ark.settings as settings
from ark.analysis import dimensionality_reduction
import test_utils


def test_plot_dim_reduced_data():
    # this only tests errors, test_dimensionality_reduction tests the meat of this function
    random_cell_data = test_utils.make_cell_table(300)

    with pytest.raises(FileNotFoundError):
        # trying to save to a non-existant directory
        dimensionality_reduction.plot_dim_reduced_data(component_one=random_cell_data.iloc[:, 0],
                                                       component_two=random_cell_data.iloc[:, 1],
                                                       fig_id=1,
                                                       hue=random_cell_data.iloc[:, 2],
                                                       cell_data=random_cell_data,
                                                       title="Title",
                                                       save_dir="bad_dir")

    with pytest.raises(FileNotFoundError):
        # setting save_dir but not setting save_file
        dimensionality_reduction.plot_dim_reduced_data(component_one=random_cell_data.iloc[:, 0],
                                                       component_two=random_cell_data.iloc[:, 1],
                                                       fig_id=1,
                                                       hue=random_cell_data.iloc[:, 2],
                                                       cell_data=random_cell_data,
                                                       title="Title",
                                                       save_dir=".")


def test_dimensionality_reduction():
    random_cell_data = test_utils.make_cell_table(300)
    test_cols = test_utils.TEST_MARKERS

    test_algorithms = ['PCA', 'tSNE', 'UMAP']

    with pytest.raises(ValueError):
        # trying to specify an algorithm not in test_algorithms
        dimensionality_reduction.visualize_dimensionality_reduction(random_cell_data,
                                                                    test_cols,
                                                                    settings.CELL_TYPE,
                                                                    algorithm="bad_alg")

    with tempfile.TemporaryDirectory() as temp_dir:
        for alg in test_algorithms:
            # test without saving, assert that the path does not exist
            dimensionality_reduction.visualize_dimensionality_reduction(random_cell_data,
                                                                        test_cols,
                                                                        settings.CELL_TYPE,
                                                                        algorithm=alg)
            assert not os.path.exists(os.path.join(temp_dir, alg + 'Visualization.png'))

            # test with saving, assert that the path does exist
            dimensionality_reduction.visualize_dimensionality_reduction(random_cell_data,
                                                                        test_cols,
                                                                        settings.CELL_TYPE,
                                                                        algorithm=alg,
                                                                        save_dir=temp_dir)
            assert os.path.exists(os.path.join(temp_dir, alg + 'Visualization.png'))
