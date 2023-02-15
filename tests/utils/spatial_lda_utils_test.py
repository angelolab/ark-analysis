import copy
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

import ark.settings as settings
import ark.spLDA.processing as pros
import ark.utils.spatial_lda_utils as spu
from test_utils import make_cell_table


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
    all_clusters = list(np.unique(cell_table[settings.CELL_TYPE]))
    formatted_table = pros.format_cell_table(cell_table, clusters=all_clusters)
    featurized_table = pros.featurize_cell_table(formatted_table)
    k_means = KMeans(n_clusters=5).fit(featurized_table["featurized_fovs"])
    wk = spu.within_cluster_sums(featurized_table["featurized_fovs"], k_means.labels_)
    # check for strictly positive value
    assert wk >= 0


def test_make_plot_fn():
    with pytest.raises(ValueError, match="Must provide difference_matrices"):
        spu.make_plot_fn(plot="adjacency")
    with pytest.raises(ValueError, match="Must provide cell_table and topic_weights"):
        spu.make_plot_fn(plot="topic_assignment")


def test_plot_topics_heatmap():
    topics = np.array([[1, 2, 3], [2, 3, 4]])
    features = ['f1', 'f2', 'f3']
    spu.plot_topics_heatmap(topics, features)


def test_plot_fovs_with_topics():
    fig, ax = plt.subplots(1, 1)
    valid_cell_table = {1: pd.DataFrame([[1, 1, 1], ], columns=["x", "y", "is_index"])}
    topic_weights = pd.DataFrame([1], index=[(1, 0)], columns=["Topic-1"])
    spu.plot_fovs_with_topics(ax, 1, topic_weights, valid_cell_table)


def test_save_spatial_lda_data():
    cell_table = make_cell_table(num_cells=1000)
    all_clusters = list(np.unique(cell_table[settings.CELL_TYPE]))
    cell_table_format = pros.format_cell_table(cell_table, clusters=all_clusters)
    # test for non-existent directory
    with pytest.raises(ValueError, match="'dir' must be a valid directory."):
        # trying to save on a non-existant directory
        spu.save_spatial_lda_file(data=cell_table_format, dir="bad_dir", file_name="file",
                                  format="pkl")

    with tempfile.TemporaryDirectory() as temp_dir:
        # test for valid format
        with pytest.raises(ValueError, match="format must be"):
            spu.save_spatial_lda_file(data=cell_table_format, dir=temp_dir, file_name="file",
                                      format="bad")
        # test that pkl format saves correctly
        spu.save_spatial_lda_file(data=cell_table_format, dir=temp_dir, file_name="file",
                                  format="pkl")
        assert os.path.exists(os.path.join(temp_dir, "file.pkl"))
        # test that csv format saves correctly
        spu.save_spatial_lda_file(data=cell_table_format[1], dir=temp_dir, file_name="file",
                                  format="csv")
        assert os.path.exists(os.path.join(temp_dir, "file.pkl"))
        # test for incorrect type and format match
        with pytest.raises(ValueError, match="'data' is of type"):
            spu.save_spatial_lda_file(data=cell_table_format, dir=temp_dir, file_name="dict",
                                      format="csv")


def test_read_spatial_lda_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "fake_file.txt")
        with open(file_path, "w") as f:
            f.write("content")
        with pytest.raises(FileNotFoundError):
            spu.read_spatial_lda_file(dir=temp_dir, file_name="bad_file")
        with pytest.raises(ValueError, match="format must be either"):
            spu.read_spatial_lda_file(dir=temp_dir, file_name="fake_file", format="txt")
