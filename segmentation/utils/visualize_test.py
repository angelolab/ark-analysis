import numpy as np
import random
from segmentation.utils import visualize
import importlib
import pandas as pd
import string

importlib.reload(visualize)


def test_visualize_z_scores():
    # Create random Z score
    z = np.random.uniform(low=-5, high=5, size=(26, 26))
    # Assign random phenotype titles
    pheno_titles = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    plot = visualize.visualize_z_scores(z, pheno_titles)


def test_get_sort_data():
    rand_type = random.choices(string.ascii_lowercase, k=100)
    ids = random.choices(range(1,10), k=100)
    random_data = pd.DataFrame.from_dict({"PatientID": ids, "cell_type": rand_type})
    print("Before sorting")
    print(random_data)
    print("----------------------------------")
    print("After sorting")
    print(visualize.get_sorted_data(random_data, "PatientID", "cell_type"))
    

def test_visualize_cells():

    rand_type = random.choices(string.ascii_lowercase, k=100)
    ids = random.choices(range(1, 3), k=100)
    rand_dict = {"PatientID": ids, "cell_type": rand_type}
    df = pd.DataFrame.from_dict(rand_dict)
    visualize.visualize_patient_population_distribution(df, "PatientID", "cell_type")
