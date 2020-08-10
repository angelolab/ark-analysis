import numpy as np
import random
from segmentation.utils import visualize
import importlib
import pandas as pd
import string
import os.path
from os import path

importlib.reload(visualize)


def test_visualize_z_scores():
    # Create random Z score
    z = np.random.uniform(low=-5, high=5, size=(26, 26))
    # Assign random phenotype titles
    pheno_titles = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    plot = visualize.visualize_z_scores(z, pheno_titles)


def test_get_sort_data():
    rand_type = random.choices(string.ascii_lowercase, k=100)
    ids = random.choices(range(1, 10), k=100)
    random_data = pd.DataFrame.from_dict({"PatientID": ids, "cell_type": rand_type})
    sorted_data = visualize.get_sorted_data(random_data, "PatientID", "cell_type")

    row_sums = [row.sum() for index, row in sorted_data.iterrows()]
    assert sorted(row_sums)


def test_visualize_cells():
    rand_type = random.choices(string.ascii_lowercase, k=100)
    ids = random.choices(range(1, 3), k=100)
    df = pd.DataFrame.from_dict({"PatientID": ids, "cell_type": rand_type})
    visualize.visualize_patient_population_distribution(df, "PatientID", "cell_type", save_dir="")

    # Check if correct plots are saved
    assert path.exists("PopulationDistribution.png")
    assert path.exists("TotalPopulationDistribution.png")
    assert path.exists("PopulationProportion.png")
