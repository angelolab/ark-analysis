import os
import numpy as np

from ark.analysis import visualize
from ark.utils import test_utils


def test_draw_boxplot():
    # trim random data so we don't have to visualize as many facets
    random_data = test_utils.make_segmented_csv(100)
    random_data = random_data[random_data['PatientID'].isin(np.arange(1, 5))]

    # most basic visualization: just data and a column name
    visualize.draw_boxplot(cell_data=random_data, col_name="A", save_dir=".")
    assert os.path.exists("sample_boxplot_viz.png")
    os.remove("sample_boxplot_viz.png")

    # next level up: data, a column name, and a split column
    visualize.draw_boxplot(cell_data=random_data, col_name="A",
                           col_split="PatientID", save_dir=".")
    assert os.path.exists("sample_boxplot_viz.png")
    os.remove("sample_boxplot_viz.png")

    # highest level: data, a column name, a split column, and split vals
    visualize.draw_boxplot(cell_data=random_data, col_name="A",
                           col_split="PatientID", split_vals=[1, 2],
                           save_dir=".")
    assert os.path.exists("sample_boxplot_viz.png")
    os.remove("sample_boxplot_viz.png")


def test_visualize_z_scores():
    # Create random Z score
    z = np.random.uniform(low=-5, high=5, size=(26, 26))
    # Assign random phenotype titles
    pheno_titles = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    visualize.visualize_z_scores(z, pheno_titles)


def test_get_sort_data():
    random_data = test_utils.make_segmented_csv(100)
    sorted_data = visualize.get_sorted_data(random_data, "PatientID", "cell_type")

    row_sums = [row.sum() for index, row in sorted_data.iterrows()]
    assert list(reversed(row_sums)) == sorted(row_sums)


def test_visualize_cells():
    random_data = test_utils.make_segmented_csv(100)
    visualize.visualize_patient_population_distribution(random_data, "PatientID", "cell_type",
                                                        save_dir=".")

    # Check if correct plots are saved
    assert os.path.exists("PopulationDistribution.png")
    assert os.path.exists("TotalPopulationDistribution.png")
    assert os.path.exists("PopulationProportion.png")
