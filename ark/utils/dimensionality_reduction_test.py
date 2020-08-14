import pandas as pd
import numpy as np
import string
import random as rand
from segmentation.utils import dimensionality_reduction
import importlib
import tempfile
import os

importlib.reload(dimensionality_reduction)


def test_dimensionality_reduction():
    random_cell_data = pd.DataFrame(np.random.random(size=(300, 8)), columns=list('ABCDEFGH'))
    random_cell_data["cell_type"] = rand.choices(string.ascii_lowercase, k=300)

    test_algorithms = ['PCA', 'tSNE', 'UMAP']
    with tempfile.TemporaryDirectory() as temp_dir:
        for alg in test_algorithms:
            dimensionality_reduction.visualize_dimensionality_reduction(random_cell_data,
                                                                        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                                                                        "cell_type",
                                                                        algorithm=alg)
            assert not os.path.exists(os.path.join(temp_dir, alg + 'Visualization.png'))

        for alg in test_algorithms:
            dimensionality_reduction.visualize_dimensionality_reduction(random_cell_data,
                                                                        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                                                                        "cell_type",
                                                                        algorithm=alg, save_dir=temp_dir)
            assert os.path.exists(os.path.join(temp_dir, alg + 'Visualization.png'))
