import os
import numpy as np
import xarray as xr
import pytest
import tempfile

from ark.analysis import visualize
from ark.utils import test_utils

import ark.settings as settings
import timeit


def test_draw_heatmap():
    # Create random Z score
    z = np.random.uniform(low=-5, high=5, size=(26, 26))
    # Assign random phenotype titles
    pheno_titles = [chr(i) for i in range(ord('a'), ord('z') + 1)]

    with pytest.raises(FileNotFoundError):
        # trying to save on a non-existant directory
        visualize.draw_heatmap(z, pheno_titles, pheno_titles, save_dir="bad_dir")

    # most basic visualization with just data, xlabels, ylabels
    with tempfile.TemporaryDirectory() as temp_dir:
        # test that without save_dir, we do not save
        visualize.draw_heatmap(z, pheno_titles, pheno_titles)
        assert not os.path.exists(os.path.join(temp_dir, "z_score_viz.png"))

        # test that with save_dir, we do save
        visualize.draw_heatmap(z, pheno_titles, pheno_titles,
                               save_dir=temp_dir)
        assert os.path.exists(os.path.join(temp_dir, "z_score_viz.png"))


def test_draw_boxplot():
    # trim random data so we don't have to visualize as many facets
    start_time = timeit.default_timer()
    random_data = test_utils.make_segmented_csv(100)
    random_data = random_data[random_data[settings.PATIENT_ID].isin(np.arange(1, 5))]

    # basic error testing
    with pytest.raises(ValueError):
        # non-existant col_name
        visualize.draw_boxplot(cell_data=random_data, col_name="AA")

    with pytest.raises(ValueError):
        # split_vals specified but not col_split
        visualize.draw_boxplot(cell_data=random_data, col_name="A", split_vals=[])

    with pytest.raises(ValueError):
        # non-existant col_split specified
        visualize.draw_boxplot(cell_data=random_data, col_name="A", col_split="AA")

    with pytest.raises(ValueError):
        # split_vals not found in col_split found
        visualize.draw_boxplot(cell_data=random_data, col_name="A",
                               col_split=settings.PATIENT_ID, split_vals=[3, 4, 5, 6])

    with pytest.raises(FileNotFoundError):
        # trying to save to a non-existant directory
        visualize.draw_boxplot(cell_data=random_data, col_name="A",
                               save_dir="bad_dir")

    # highest level: data, a column name, a split column, and split vals
    with tempfile.TemporaryDirectory() as temp_dir:
        visualize.draw_boxplot(cell_data=random_data, col_name="A",
                               col_split=settings.PATIENT_ID, split_vals=[1, 2],
                               save_dir=temp_dir)
        assert os.path.exists(os.path.join(temp_dir, "boxplot_viz.png"))


def test_get_sort_data():
    random_data = test_utils.make_segmented_csv(100)
    sorted_data = visualize.get_sorted_data(random_data, settings.PATIENT_ID, settings.CELL_TYPE)

    row_sums = [row.sum() for index, row in sorted_data.iterrows()]
    assert list(reversed(row_sums)) == sorted(row_sums)


def test_plot_barchart():
    # mostly error checking here, test_visualize_cells tests the meat of the functionality
    random_data = test_utils.make_segmented_csv(100)

    with pytest.raises(FileNotFoundError):
        # trying to save to a non-existant directory
        visualize.plot_barchart(random_data, "Random Title", "Random X Label",
                                "Random Y Label", save_dir="bad_dir")

    with pytest.raises(FileNotFoundError):
        # setting save_dir but not setting save_file
        visualize.plot_barchart(random_data, "Random Title", "Random X Label",
                                "Random Y Label", save_dir=".")


def test_visualize_patient_population_distribution():
    random_data = test_utils.make_segmented_csv(100)

    with tempfile.TemporaryDirectory() as temp_dir:
        # test without a save_dir, check that we do not save the files
        visualize.visualize_patient_population_distribution(random_data, settings.PATIENT_ID,
                                                            settings.CELL_TYPE)

        assert not os.path.exists(os.path.join(temp_dir, "PopulationDistribution.png"))
        assert not os.path.exists(os.path.join(temp_dir, "TotalPopulationDistribution.png"))
        assert not os.path.exists(os.path.join(temp_dir, "PopulationProportion.png"))

        # now test with a save_dir, which will check that we do save the files
        visualize.visualize_patient_population_distribution(random_data, settings.PATIENT_ID,
                                                            settings.CELL_TYPE, save_dir=temp_dir)

        # Check if correct plots are saved
        assert os.path.exists(os.path.join(temp_dir, "PopulationDistribution.png"))
        assert os.path.exists(os.path.join(temp_dir, "TotalPopulationDistribution.png"))
        assert os.path.exists(os.path.join(temp_dir, "PopulationProportion.png"))


def test_visualize_neighbor_cluster_metrics():
    # create the random cluster scores xarray
    random_cluster_stats = np.random.uniform(low=0, high=100, size=9)
    random_coords = [np.arange(2, 11)]
    random_dims = ["cluster_num"]
    random_data = xr.DataArray(random_cluster_stats, coords=random_coords,
                               dims=random_dims)

    # error checking
    with pytest.raises(FileNotFoundError):
        # specifying a non-existent directory to save to
        visualize.visualize_neighbor_cluster_metrics(random_data, save_dir="bad_dir")

    with tempfile.TemporaryDirectory() as temp_dir:
        # test that without save_dir, we do not save
        visualize.visualize_neighbor_cluster_metrics(random_data)
        assert not os.path.exists(os.path.join(temp_dir, "neighborhood_cluster_scores.png"))

        # test that with save_dir, we do save
        visualize.visualize_neighbor_cluster_metrics(random_data, save_dir=temp_dir)
        assert os.path.exists(os.path.join(temp_dir, "neighborhood_cluster_scores.png"))
