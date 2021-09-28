import io
from pathlib import Path
from tempfile import tempdir

import numpy as np
import pandas as pd
import pytest

from .metaclusterdata import MetaClusterData, metaclusterdata_from_files

THIS_DIR = Path(__file__).parent
TESTDATA_DIR = THIS_DIR / 'test_data'
DATA_DIR = THIS_DIR.parent.parent.parent / 'data'
MC_DATA_DIR = DATA_DIR / "example_dataset" / "metaclustering"


def as_csv(df):
    """Returns an in-memory csv of Pandas.DataFrame"""
    f = io.StringIO()
    df.to_csv(f, index=False)
    f.seek(0)
    return f


def test_can_read_csvs(simple_clusters_df, simple_pixelcount_df):
    metaclusterdata_from_files(as_csv(simple_clusters_df), as_csv(simple_pixelcount_df))


def test_requires_cluster_column():
    with pytest.raises(AssertionError):
        metaclusterdata_from_files(
            TESTDATA_DIR / "ex1_clusters_nozscore_nocluster.csv",
            MC_DATA_DIR / "ex1_clusters_pixelcount.csv",
            metacluster_header='hCluster_cap')


def test_requires_metacluster_column():
    with pytest.raises(AssertionError):
        metaclusterdata_from_files(
            MC_DATA_DIR / "ex1_clusters_nozscore.csv",
            MC_DATA_DIR / "ex1_clusters_pixelcount.csv")


def test_requires_rows_match():
    with pytest.raises(AssertionError):
        metaclusterdata_from_files(
            MC_DATA_DIR / "ex1_clusters_nozscore.csv",
            TESTDATA_DIR / "ex1_clusters_pixelcount_mismatchedids.csv",
            metacluster_header='hCluster_cap')


def test_requires_rows_match():
    with pytest.raises(AssertionError):
        metaclusterdata_from_files(
            TESTDATA_DIR / "ex1_clusters_nozscore_notint.csv",
            MC_DATA_DIR / "ex1_clusters_pixelcount.csv",
            metacluster_header='hCluster_cap')


def test_requires_unique_clusterid():
    with pytest.raises(AssertionError):
        metaclusterdata_from_files(
            TESTDATA_DIR / "ex1_clusters_nozscore_notuniqueid.csv",
            MC_DATA_DIR / "ex1_clusters_pixelcount.csv",
            metacluster_header='hCluster_cap')


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
        "cluster,metacluster,mc_name",
        "1,1,1",
        "2,2,2",
        "3,3,3",
        "4,3,3",
        ]


def test_metaclusters_can_have_displaynames(simple_metaclusterdata: MetaClusterData):
    assert simple_metaclusterdata.metacluster_displaynames == ['1', '2', '3']


def test_metaclusters_can_change_displaynames(simple_metaclusterdata: MetaClusterData):
    simple_metaclusterdata.change_displayname(1, 'y2k')
    assert simple_metaclusterdata.metacluster_displaynames == ['y2k', '2', '3']


def test_can_find_which_metacluster_a_cluster_belongs_to(simple_metaclusterdata: MetaClusterData):
    assert simple_metaclusterdata.which_metacluster(4) == 3


def test_can_average_clusters_by_metacluster(simple_metaclusterdata: MetaClusterData):
    simple_metaclusterdata.remap(4, 3)
    clusters_data = np.array([
        (0.1, 0.2, 0.1),
        (0.1, 0.1, 0.3),
        ((0.5*50+0.7*77)/(50+77),
         (0.1*50+0.2*77)/(50+77),
         (0.1*50+0.1*77)/(50+77)),
        ])
    np.testing.assert_equal(simple_metaclusterdata.metaclusters.values, clusters_data)


def test_can_reorder_markers(simple_metaclusterdata: MetaClusterData):
    print(simple_metaclusterdata._marker_order)
    simple_metaclusterdata.set_marker_order([0, 2, 1])
    print(simple_metaclusterdata._marker_order)
    assert list(simple_metaclusterdata.marker_names) == ['CD163', 'CD31', 'CD206']
