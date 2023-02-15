import io
import os
import tempfile

import pytest

from ark.utils.metacluster_remap_gui import metaclusterdata_from_files


def as_csv(df):
    """Returns an in-memory csv of Pandas.DataFrame"""
    f = io.StringIO()
    df.to_csv(f, index=False)
    f.seek(0)
    return f


def test_can_read_csvs(simple_full_cluster_data):
    md = metaclusterdata_from_files(as_csv(simple_full_cluster_data))
    assert list(md.cluster_pixelcounts.columns.values) == ['count']


def test_can_read_csvs_prefix_trim(simple_full_cluster_data):
    simple_full_cluster_data.rename(columns={'count': 'prefix_count'}, inplace=True)

    md = metaclusterdata_from_files(as_csv(simple_full_cluster_data), prefix_trim='prefix_')
    assert list(md.cluster_pixelcounts.columns.values) == ['count']


def test_requires_valid_path(simple_full_cluster_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        simple_full_cluster_data.to_csv(temp_dir + '/sample.csv', index=False)

        with pytest.raises(FileNotFoundError):
            metaclusterdata_from_files(os.path.join(temp_dir, 'bad_sample.csv'))


def test_requires_valid_cluster_type(simple_full_cluster_data):
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_full_cluster_data), 'bad_cluster_type')


def test_pixelcounts_requires_cluster_column(simple_full_cluster_data):
    simple_full_cluster_data.rename(columns={'cluster': 'wrongname'}, inplace=True)
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_full_cluster_data))


def test_pixelcounts_requires_count_column(simple_full_cluster_data):
    simple_full_cluster_data.rename(columns={'count': 'wrongname'}, inplace=True)
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_full_cluster_data))


def test_requires_metacluster_column(simple_full_cluster_data):
    simple_full_cluster_data.rename(columns={'metacluster': 'wrongname'}, inplace=True)
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_full_cluster_data))


def test_requires_int_cluster_ids(simple_full_cluster_data):
    simple_full_cluster_data = simple_clusters_data.astype({'cluster': str})
    simple_full_cluster_data.at[1, 'cluster'] = 'd'
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_full_cluster_data))


def test_requires_unique_clusterid(simple_full_cluster_data):
    simple_full_cluster_data.at[1, 'cluster'] = 3
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_full_cluster_data))


def test_requires_int_cluster_ids(simple_full_cluster_data):
    simple_full_cluster_data['cluster'] = [
        f"X{id}" for id in simple_full_cluster_data['cluster'].values
    ]
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_full_cluster_data))


def test_requires_ids_to_not_start_at_0(simple_full_cluster_data):
    simple_full_cluster_data['cluster'] = [
        id - 1 for id in simple_full_cluster_data['cluster'].values
    ]
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_full_cluster_data))
