import numpy as np
from segmentation.utils import visualize
import importlib
importlib.reload(visualize)


def test_visualize_z_scores():
    # Create random Z score
    z = np.random.uniform(low=-5, high=5, size=(26, 26))
    # Assign random phenotype titles
    pheno_titles = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    plot = visualize.visualize_z_scores(z, pheno_titles)

