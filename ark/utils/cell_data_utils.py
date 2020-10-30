import os
import numpy as np
import pandas as pd

import ark.settings as settings


def append_cell_lineage_col(exp_data, lineage_info, cell_type_col="cell_type",
                            cell_lin_col=settings.CELL_LINEAGE):
    """Given a dictionary defining which cell types correspond to which
    cell lineages, create a column in exp_data that explicitly labels
    each cell with their respective lineage. Especially useful as a
    preprocessing step before context-dependent spatial analysis.

    Args:
        exp_data (pandas.DataFrame):
            A dataframe defining the properties of each cell
        lineage_info (dict):
            A dictionary that maps cell lineages to a list of cell types that
            should correspond to that lineage
        cell_type_col (str):
            The column name defining cell type in exp_data
        cell_lin_col (str):
            The name of the cell lineage column desired. Any cell type not
            appearing in a list in lineage_info will be removed from exp_data

    Returns:
        pandas.DataArray:
            An updated expression matrix with a new column (cell_lin_col)
            with the proper cell lineage labels
    """

    # the cell_type_col specified has to actually exist in exp_data
    if cell_type_col not in exp_data.columns.values:
        raise ValueError("cell_type_col %s does not exist in expression_data" % cell_type_col)

    cell_types = []
    for lin_list in lineage_info.values():
        cell_types = cell_types + lin_list

    # do not allow the user to specify multiple cell types across the list values of lineage_info
    if len(set(cell_types)) < len(cell_types):
        raise ValueError("Duplicate cell types specified in lineage_info")

    # cannot specify non-existant cell types in expression data
    if not np.all(np.isin(cell_types, exp_data[cell_type_col].values)):
        raise ValueError("Some cell types in lineage info do not exist in cell_type_col")

    # only keep the rows that have cell types contained in a list in lineage_info
    exp_data_with_lin = exp_data[exp_data[cell_type_col].isin(cell_types)].copy()

    # create the cell_lin_col and assign the proper lineage labels to them
    exp_data[cell_lin_col] = ''
    for lin, lin_list in lineage_info.items():
        exp_data_with_lin.loc[exp_data_with_lin[cell_type_col].isin(lin_list), cell_lin_col] = lin

    return exp_data_with_lin
