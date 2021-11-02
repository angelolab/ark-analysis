import io
from tempfile import tempdir

import pytest

from . import metaclusterdata_from_files


def as_csv(df):
    """Returns an in-memory csv of Pandas.DataFrame"""
    f = io.StringIO()
    df.to_csv(f, index=False)
    f.seek(0)
    return f


def test_can_read_csvs(simple_clusters_df, simple_pixelcount_df):
    metaclusterdata_from_files(as_csv(simple_clusters_df), as_csv(simple_pixelcount_df))


def test_requires_cluster_column(simple_clusters_df, simple_pixelcount_df):
    simple_clusters_df.rename(columns={'cluster': 'wrongname'}, inplace=True)
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_clusters_df), as_csv(simple_pixelcount_df))


def test_pixelcounts_requires_cluster_column(simple_clusters_df, simple_pixelcount_df):
    simple_pixelcount_df.rename(columns={'cluster': 'wrongname'}, inplace=True)
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_clusters_df), as_csv(simple_pixelcount_df))


def test_pixelcounts_requires_count_column(simple_clusters_df, simple_pixelcount_df):
    simple_pixelcount_df.rename(columns={'count': 'wrongname'}, inplace=True)
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_clusters_df), as_csv(simple_pixelcount_df))


def test_requires_metacluster_column(simple_clusters_df, simple_pixelcount_df):
    simple_clusters_df.rename(columns={'metacluster': 'wrongname'}, inplace=True)
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_clusters_df), as_csv(simple_pixelcount_df))


def test_requires_rows_match(simple_clusters_df, simple_pixelcount_df):
    simple_pixelcount_df.drop(1, inplace=True)
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_clusters_df), as_csv(simple_pixelcount_df))


def test_requires_int_cluster_ids(simple_clusters_df, simple_pixelcount_df):
    simple_clusters_df = simple_clusters_df.astype({'cluster': str})
    simple_clusters_df.at[1, 'cluster'] = 'd'
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_clusters_df), as_csv(simple_pixelcount_df))


def test_requires_unique_clusterid(simple_clusters_df, simple_pixelcount_df):
    simple_clusters_df.at[1, 'cluster'] = 3
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_clusters_df), as_csv(simple_pixelcount_df))


def test_requires_int_cluster_ids(simple_clusters_df, simple_pixelcount_df):
    simple_clusters_df['cluster'] = [f"X{id}" for id in simple_clusters_df['cluster'].values]
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_clusters_df), as_csv(simple_pixelcount_df))


def test_requires_ids_to_not_start_at_0(simple_clusters_df, simple_pixelcount_df):
    simple_clusters_df['cluster'] = [id - 1 for id in simple_clusters_df['cluster'].values]
    with pytest.raises(ValueError):
        metaclusterdata_from_files(as_csv(simple_clusters_df), as_csv(simple_pixelcount_df))
