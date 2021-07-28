import os
import numpy as np
import pandas as pd
import tempfile
import pytest
from ark.utils.spatial_lda_utils import check_format_cell_table_args, check_featurize_cell_table_args
from ark.settings import BASE_COLS

def test_check_format_cell_table_args():
    # Testing variables
    for i in ["Au", "CD4", "CD8"]:
        BASE_COLS.append(i)

    # Cell table pd.DataFrame
    VALID_DF = pd.DataFrame(columns=BASE_COLS)
    # Doesn't meet minimum column requirements
    INVALID_DF1 = pd.DataFrame(columns=BASE_COLS[1:6])

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

    # DataFrame Checks
    with pytest.raises(ValueError):
        check_format_cell_table_args(INVALID_DF1, VALID_MARKERS, VALID_CLUSTERS, VALID_FOVS)
    # Markers/Clusters Checks
    with pytest.raises(ValueError):
        check_format_cell_table_args(VALID_DF, None, None, VALID_FOVS)
    with pytest.raises(TypeError):
        check_format_cell_table_args(VALID_DF, 1, None, VALID_FOVS)
    with pytest.raises(ValueError):
        check_format_cell_table_args(VALID_DF, INVALID_MARKERS1, None, VALID_FOVS)
    with pytest.raises(TypeError):
        check_format_cell_table_args(VALID_DF, INVALID_MARKERS2, None, VALID_FOVS)
    with pytest.raises(ValueError):
        check_format_cell_table_args(VALID_DF, INVALID_MARKERS3, None, VALID_FOVS)
    with pytest.raises(TypeError):
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, 1, VALID_FOVS)
    with pytest.raises(TypeError):
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, INVALID_CLUSTERS1, VALID_FOVS)
    with pytest.raises(ValueError):
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, INVALID_CLUSTERS2, VALID_FOVS)
    # FOV Checks
    with pytest.raises(TypeError):
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, VALID_CLUSTERS, 1)
    with pytest.raises(ValueError):
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, VALID_CLUSTERS, INVALID_FOVS1)
    with pytest.raises(ValueError):
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, VALID_CLUSTERS, INVALID_FOVS2)
