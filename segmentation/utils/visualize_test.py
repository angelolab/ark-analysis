import numpy as np
import random
from segmentation.utils import visualize
import importlib
importlib.reload(visualize)


def test_visualize_z_scores():
    # Create random Z score
    z = np.random.uniform(low=-5, high=5, size=(26, 26))
    # Assign random phenotype titles
    pheno_titles = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    plot = visualize.visualize_z_scores(z, pheno_titles)
    
def test_visualize_cells():
    rand_type = []
    ids = []
    for x in range(0,1000):
        rand_type.append(chr(random.randint(0,25) + 97))
        ids.append(random.randint(1,5))
    print(rand_type, ids)
    rand_dict = {"PatientID": ids, "cell_type": rand_type}
    df = pd.DataFrame.from_dict(rand_dict)
    print(df)
    visualize_cell_distribution_in_all_patients(df, "cell_type")
    visualize_distribution_of_cell_count(df, "PatientID", "cell_type")
    visualize_proportion_of_cell_count(df, "PatientID", "cell_type")
