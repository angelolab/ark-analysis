import numpy as np
from os import path

from ark.analysis import visualize
from ark.utils import test_utils


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
                                                        save_dir="")

    # Check if correct plots are saved
    assert path.exists("PopulationDistribution.png")
    assert path.exists("TotalPopulationDistribution.png")
    assert path.exists("PopulationProportion.png")
