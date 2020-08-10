import pandas as pd
import numpy as np
import string
import random as rand
from segmentation.utils import dimensionality_reduction
import importlib

importlib.reload(dimensionality_reduction)


def test_dimensionality_reduction():
    random_cell_data = pd.DataFrame(np.random.random(size=(300, 8)), columns=list('ABCDEFGH'))
    random_cell_data["cell_type"] = rand.choices(string.ascii_lowercase, k=300)

    test_algorithms = ['PCA', 'tSNE', 'UMAP']
    for alg in test_algorithms:
        dimensionality_reduction.visualize_dimensionality_reduction(random_cell_data,
                                                                    ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                                                                    "cell_type",
                                                                    algorithm=alg, save_dir="")
