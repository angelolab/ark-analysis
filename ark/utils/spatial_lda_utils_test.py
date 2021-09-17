import pandas as pd
import numpy as np
import pytest
import copy

from sklearn.cluster import KMeans
from ark.utils.test_utils import make_cell_table
from ark.spLDA.processing import format_cell_table, featurize_cell_table
import ark.settings as settings
import ark.utils.spatial_lda_utils as spu


def test_check_format_cell_table_args():
    # Testing variables
    cols = copy.deepcopy(settings.BASE_COLS)
    for i in ["Au", "CD4", "CD8"]:
        cols.append(i)

    # Cell table pd.DataFrame
    valid_df = pd.DataFrame(columns=cols)
    # Doesn't meet minimum column requirements
    invalid_df1 = pd.DataFrame(columns=cols[1:6])

    # Markers
    valid_markers = ["Au", "CD4", "CD8"]
    # Specifies marker not included in cell table
    invalid_markers1 = ["Au", "CD4", "CD8", "Vimentin"]
    # Includes integer
    invalid_markers2 = ["Au", "CD4", "CD8", 3]
    # Empty list
    invalid_markers3 = []

    # Cluster
    valid_clusters = [1, 2, 3]
    # Strings instead of integers
    invalid_clusters1 = ["a", "b", "c"]
    # Empty List
    invalid_clusters2 = []

    # DataFrame Checks
    with pytest.raises(ValueError):
        spu.check_format_cell_table_args(invalid_df1, valid_markers, valid_clusters)
    # Markers/Clusters Checks
    with pytest.raises(ValueError, match=r"cannot both be None"):
        spu.check_format_cell_table_args(valid_df, None, None)
    with pytest.raises(ValueError):
        spu.check_format_cell_table_args(valid_df, invalid_markers1, None)
    with pytest.raises(ValueError):
        spu.check_format_cell_table_args(valid_df, invalid_markers2, None)
    with pytest.raises(ValueError, match=r"List arguments cannot be empty"):
        spu.check_format_cell_table_args(valid_df, invalid_markers3, None)
    with pytest.raises(ValueError):
        spu.check_format_cell_table_args(valid_df, valid_markers, invalid_clusters1)
    with pytest.raises(ValueError, match=r"List arguments cannot be empty"):
        spu.check_format_cell_table_args(valid_df, valid_markers, invalid_clusters2)


def test_check_featurize_cell_table_args():
    # Testing variables
    valid_cell_table = {1: pd.DataFrame(columns=["CD4", "CD8", "is_index"])}

    valid_feature = "marker"
    invalid_feature1 = "avg_cluster"
    invalid_feature2 = 2

    valid_radius = 100
    invalid_radius1 = 20
    invalid_radius2 = "25"

    valid_cell_index = "is_index"
    invalid_cell_index1 = 1
    invalid_cell_index2 = "is_tumor"

    with pytest.raises(ValueError):
        spu.check_featurize_cell_table_args(valid_cell_table, invalid_feature1, valid_radius,
                                        valid_cell_index)
    with pytest.raises(ValueError):
        spu.check_featurize_cell_table_args(valid_cell_table, invalid_feature2, valid_radius,
                                        valid_cell_index)
    with pytest.raises(ValueError, match=r"radius must not be less than 25"):
        spu.check_featurize_cell_table_args(valid_cell_table, valid_feature, invalid_radius1,
                                        valid_cell_index)
    with pytest.raises(TypeError, match=r"radius should be of type 'int'"):
        spu.check_featurize_cell_table_args(valid_cell_table, valid_feature, invalid_radius2,
                                        valid_cell_index)
    with pytest.raises(ValueError):
        spu.check_featurize_cell_table_args(valid_cell_table, valid_feature, valid_radius,
                                        invalid_cell_index1)
    with pytest.raises(ValueError):
        spu.check_featurize_cell_table_args(valid_cell_table, valid_feature, valid_radius,
                                        invalid_cell_index2)

def test_within_cluster_sums():
    cell_table = make_cell_table(num_cells=1000)
    all_clusters = list(np.unique(cell_table[settings.CLUSTER_ID]))
    formatted_table = format_cell_table(cell_table, clusters=all_clusters)
    featurized_table = featurize_cell_table(formatted_table)
    k_means = KMeans(n_clusters=5).fit(featurized_table["featurized_fovs"])
    wk = spu.within_cluster_sums(featurized_table["featurized_fovs"], k_means.labels_)
    # check for correct type and strictly positive value
    assert type(wk) == float
    assert wk >= 0
