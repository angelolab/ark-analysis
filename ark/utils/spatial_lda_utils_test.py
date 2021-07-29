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

    # DataFrame Checks
    with pytest.raises(ValueError, match=r"table must contain"):
        check_format_cell_table_args(INVALID_DF1, VALID_MARKERS, VALID_CLUSTERS)
    # Markers/Clusters Checks
    with pytest.raises(ValueError, match=r"cannot both be None"):
        check_format_cell_table_args(VALID_DF, None, None)
    with pytest.raises(TypeError, match=r"markers must be a list"):
        check_format_cell_table_args(VALID_DF, 1, None)
    with pytest.raises(ValueError, match=r"markers must have a column"):
        check_format_cell_table_args(VALID_DF, INVALID_MARKERS1, None)
    with pytest.raises(TypeError, match=r"markers must be a list"):
        check_format_cell_table_args(VALID_DF, INVALID_MARKERS2, None)
    with pytest.raises(ValueError, match=r"marker names cannot be empty"):
        check_format_cell_table_args(VALID_DF, INVALID_MARKERS3, None)
    with pytest.raises(TypeError, match=r"clusters must be a list"):
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, 1)
    with pytest.raises(TypeError, match=r"clusters must be a list"):
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, INVALID_CLUSTERS1)
    with pytest.raises(ValueError, match=r"cluster ids cannot be empty"):
        check_format_cell_table_args(VALID_DF, VALID_MARKERS, INVALID_CLUSTERS2)

def test_check_featurize_cell_table_args():
    # Testing variables
    VALID_CELL_TABLE = {1: pd.DataFrame(columns=["CD4", "CD8", "is_index"])}

    VALID_FEATURE_BY = "marker"
    INVALID_FEATURE_BY1 = "avg_cluster"
    INVALID_FEATURE_BY2 = 2

    VALID_RADIUS = 100
    INVALID_RADIUS1 = 20
    INVALID_RADIUS2 = "25"

    VALID_CELL_INDEX = "is_index"
    INVALID_CELL_INDEX1 = 1
    INVALID_CELL_INDEX2 = "is_tumor"

    with pytest.raises(ValueError, match=r"feature_by must be one of 'cluster', 'marker', 'avg_marker', 'count'"):
        check_featurize_cell_table_args(VALID_CELL_TABLE, INVALID_FEATURE_BY1, VALID_RADIUS, VALID_CELL_INDEX)
    with pytest.raises(TypeError, match = r"feature_by should be of type 'str'"):
        check_featurize_cell_table_args(VALID_CELL_TABLE, INVALID_FEATURE_BY2, VALID_RADIUS, VALID_CELL_INDEX)
    with pytest.raises(ValueError, match=r"radius must not be less than 25"):
        check_featurize_cell_table_args(VALID_CELL_TABLE, VALID_FEATURE_BY, INVALID_RADIUS1, VALID_CELL_INDEX)
    with pytest.raises(TypeError, match=r"radius should be of type 'int'"):
        check_featurize_cell_table_args(VALID_CELL_TABLE, VALID_FEATURE_BY, INVALID_RADIUS2, VALID_CELL_INDEX)
    with pytest.raises(TypeError, match=r"cell_index should be of type 'str'"):
        check_featurize_cell_table_args(VALID_CELL_TABLE, VALID_FEATURE_BY, VALID_RADIUS, INVALID_CELL_INDEX1)
    with pytest.raises(ValueError, match=r"cell_index must be a valid column"):
        check_featurize_cell_table_args(VALID_CELL_TABLE, VALID_FEATURE_BY, VALID_RADIUS, INVALID_CELL_INDEX2)







