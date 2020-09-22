import os
import pytest
import numpy as np
import pandas as pd

import ark.utils.cell_data_utils as cell_data_utils
import ark.utils.test_utils as test_utils


def test_append_cell_lineage_col():
    sample_cell_data = test_utils.make_segmented_csv(num_cells=20)
    sample_cell_data.loc[0:4, 'cell_type'] = 'a'
    sample_cell_data.loc[5:9, 'cell_type'] = 'b'
    sample_cell_data.loc[10:14, 'cell_type'] = 'c'
    sample_cell_data.loc[15:19, 'cell_type'] = 'd'

    # note that cell_type d will be ignored
    sample_lineage_info = {
        'lin_1': ['a', 'b'],
        'lin_2': ['c']
    }

    with pytest.raises(ValueError):
        # specifying a non-existant column for cell_type_col
        sample_cell_data_lin = cell_data_utils.append_cell_lineage_col(
            exp_data=sample_cell_data, lineage_info=sample_lineage_info,
            cell_type_col="bad_col")

    with pytest.raises(ValueError):
        # specifying duplicate cell types across various lists in lineage info dict
        bad_sample_lineage_info = {
            'lin_1': ['a', 'b'],
            'lin_2': ['b', 'c']
        }

        sample_cell_data_lin = cell_data_utils.append_cell_lineage_col(
            exp_data=sample_cell_data, lineage_info=bad_sample_lineage_info)

    with pytest.raises(ValueError):
        # specifying non-existant cell types in cell_type_col of the expression matrix
        bad_sample_lineage_info = {
            'lin_1': ['e']
        }

        sample_cell_data_lin = cell_data_utils.append_cell_lineage_col(
            exp_data=sample_cell_data, lineage_info=bad_sample_lineage_info)

    sample_cell_data_lin = cell_data_utils.append_cell_lineage_col(
        exp_data=sample_cell_data, lineage_info=sample_lineage_info)

    # because 'd' cells were chopped off, we only have 15 rows, and none should have cell_type 'd'
    assert sample_cell_data_lin.shape[0] == 15
    assert 'd' not in sample_cell_data_lin['cell_type'].values

    # we should actually have a cell_lineage column
    assert 'cell_lineage' in sample_cell_data_lin.columns.values

    # test that the cell types have been mapped to their proper cell lineages
    # as defined by sample_lineage_info
    a_indices = sample_cell_data_lin['cell_type'] == 'a'
    b_indices = sample_cell_data_lin['cell_type'] == 'b'
    c_indices = sample_cell_data_lin['cell_type'] == 'c'

    assert np.all(sample_cell_data_lin.loc[a_indices, 'cell_lineage'].values == 'lin_1')
    assert np.all(sample_cell_data_lin.loc[b_indices, 'cell_lineage'].values == 'lin_1')
    assert np.all(sample_cell_data_lin.loc[c_indices, 'cell_lineage'].values == 'lin_2')
