from pathlib import Path
from tempfile import tempdir

import pandas as pd
import numpy as np
import pytest

from .metaclusterdata import MetaClusterData, metaclusterdata_from_files

THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR.parent.parent.parent / 'data'
MC_DATA_DIR = DATA_DIR / "example_dataset" / "metaclustering"


def test_can_read_example_input_1():
    metaclusterdata_from_files(
        MC_DATA_DIR / "ex1_clusters_nozscore.csv",
        MC_DATA_DIR / "ex1_clusters_pixelcount.csv")


def test_can_read_example_input_2():
    metaclusterdata_from_files(
        MC_DATA_DIR / "ex2_clusters_nozscore.csv",
        MC_DATA_DIR / "ex2_clusters_pixelcount.csv")


@pytest.fixture
def simple_metaclusterdata():
    clusters_headers = ['CD163', 'CD206', 'CD31', 'cluster', 'metacluster']
    clusters_data = [
        (0.1, 0.2, 0.1, 1, 1),
        (0.1, 0.1, 0.3, 2, 2),
        (0.5, 0.1, 0.1, 3, 3),
        (0.7, 0.2, 0.1, 4, 3),
        ]
    clusters_raw_df = pd.DataFrame(data=clusters_data, columns=clusters_headers)

    pixelcount_headers = ['cluster', 'count']
    pixelcount_data = [
        (1, 25),
        (2, 10),
        (3, 50),
        (4, 77),
        ]
    pixelcount_df = pd.DataFrame(data=pixelcount_data, columns=pixelcount_headers)

    return MetaClusterData(clusters_raw_df, pixelcount_df)


def test_can_get_mapping(simple_metaclusterdata: MetaClusterData):
    np.testing.assert_array_equal(
        simple_metaclusterdata.mapping['metacluster'].values,
        np.array((1, 2, 3, 3)))


def test_can_remap(simple_metaclusterdata: MetaClusterData):
    simple_metaclusterdata.remap(4, 1)
    assert simple_metaclusterdata.mapping.loc[4, 'metacluster'] == 1


def test_can_create_new_metacluster(simple_metaclusterdata: MetaClusterData):
    new_mc = simple_metaclusterdata.new_metacluster()
    simple_metaclusterdata.remap(4, new_mc)
    assert simple_metaclusterdata.mapping.loc[4, 'metacluster'] == 4


def test_can_save_mapping(simple_metaclusterdata: MetaClusterData, tmp_path):
    simple_metaclusterdata.output_mapping_filename = tmp_path / 'output_mapping.csv'
    simple_metaclusterdata.save_output_mapping()
    with open(tmp_path / 'output_mapping.csv', 'r') as f:
        output = [ll.strip() for ll in f.readlines()]
    assert output == [
        "cluster,metacluster",
        "1,1",
        "2,2",
        "3,3",
        "4,3",
        ]

# ptest_can_provide_alternate_name_for_metacluster
