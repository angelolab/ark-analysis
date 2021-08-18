import numpy as np
import pytest

import ark.spLDA.processing as pros
from ark.utils.misc_utils import verify_in_list
from ark.utils.test_utils import make_cell_table

# Generate, format, and featurize a test cell table
N_CELLS = 1000
TRAIN_FRAC = 0.75
TRAIN_CELLS = TRAIN_FRAC * N_CELLS
TEST_CELL_TABLE = make_cell_table(N_CELLS)
TEST_FORMAT = pros.format_cell_table(
    cell_table=TEST_CELL_TABLE, clusters=list(np.unique(TEST_CELL_TABLE["cluster_id"])))
TEST_FEATURES = pros.featurize_cell_table(cell_table=TEST_FORMAT, train_frac=TRAIN_FRAC)


def test_format_cell_table():
    # Check that number of FOVS match
    verify_in_list(
        fovs1=list(np.unique(TEST_CELL_TABLE["SampleID"])), fovs2=list(TEST_FORMAT.keys()))
    # Check that columns were retained/renamed
    verify_in_list(
        cols1=["x", "y", "cluster_id", "cluster", "is_index"], cols2=list(TEST_FORMAT[1].columns))
    # Check that columns were dropped
    assert len(TEST_CELL_TABLE.columns) < len(TEST_FORMAT[1].columns)


def test_featurize_cell_table():
    # Check for consistent dimensions
    assert TEST_FEATURES["featurized_fovs"].shape[0] == TEST_CELL_TABLE.shape[0]
    assert TEST_FEATURES["featurized_fovs"].shape[0] == N_CELLS
    assert TEST_FEATURES["train_features"].shape[0] == TRAIN_CELLS


def test_compute_topic_eda():
    # at least 25 bootstrap iterations
    with pytest.raises(ValueError, match="Number of bootstrap samples must be at least 25"):
        pros.compute_topic_eda(TEST_FEATURES["featurized_fovs"], topics=[5], num_boots=20)
    # appropriate range of topics
    with pytest.raises(ValueError, match="Number of topics must be greater than 2"):
        pros.compute_topic_eda(TEST_FEATURES["featurized_fovs"], topics=[2], num_boots=25)
    with pytest.raises(ValueError, match=r"Number of topics must be less than"):
        pros.compute_topic_eda(TEST_FEATURES["featurized_fovs"], topics=[1000], num_boots=25)
