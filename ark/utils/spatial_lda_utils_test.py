import os
import numpy as np
import pandas as pd
import tempfile
import pytest
from ark.utils.spatial_lda_utils import *


def test_check_format_cell_table_args():
    # Testing variables
    base_cols = [
        "point",
        "label",
        "cell_size",
        "centroid-0",
        "centroid-1",
        "pixelfreq_hclust_cap",
        "name",
        "Au",
        "CD4",
        "CD8"
    ]
    # Cell table pd.DataFrame
    VALID_DF = pd.DataFrame(columns=base_cols)
    # Not a pd.DataFrame
    INVALID_DF1 = []
    # Doesn't meet minimum column requirements
    INVALID_DF2 = pd.DataFrame(columns=base_cols[1:6])

    # Markers
    VALID_MARKERS = ["Au", "CD4", "CD8"]
    # Specifies marker not included in cell table
    INVALID_MARKERS1 = ["Au", "CD4", "CD8", "Vimentin"]
    # Includes integer
    INVALID_MARKERS2 = ["Au", "CD4", "CD8", 3]
    # Empty list
    INVALID_MARKERS3 = []

    # Cluster
    VALID_CLUSTERS = [1, 2, 3]
    # Strings instead of integers
    INVALID_CLUSTERS1 = ["a", "b", "c"]
    # Empty List
    INVALID_CLUSTERS2 = []

    # FOVs
    VALID_FOVS = [1, 2, 3]
    # Strings instead of integers
    INVALID_FOVS1 = ["a", "b", "c"]
    # Empty List
    INVALID_FOVS2 = []

    # Run Checks
    with pytest.raises(ValueError):
        # DataFrame Checks
        check_format_cell_table_args(INVALID_DF1, VALID_MARKERS, VALID_CLUSTERS, VALID_FOVS)
        check_format_cell_table_args(INVALID_DF2, VALID_MARKERS, VALID_CLUSTERS, VALID_FOVS)
        # Markers/Clusters Checks
        check_format_cell_table_args(VALID_DF, None, None, VALID_FOVS)
        check_format_cell_table_args(VALID_DF, 1, None, VALID_FOVS)
        check_format_cell_table_args(VALID_DF, INVALID_MARKERS1, None, VALID_FOVS)
        check_format_cell_table_args(VALID_DF, INVALID_MARKERS2, None, VALID_FOVS)
        check_format_cell_table_args(VALID_DF, INVALID_MARKERS3, None, VALID_FOVS)
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, 1, VALID_FOVS)
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, INVALID_CLUSTERS1, VALID_FOVS)
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, INVALID_CLUSTERS2, VALID_FOVS)
        # FOV Checks
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, VALID_CLUSTERS, 1)
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, VALID_CLUSTERS, INVALID_FOVS1)
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, VALID_CLUSTERS, INVALID_FOVS2)









