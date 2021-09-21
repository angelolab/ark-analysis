from pathlib import Path

import pandas as pd
from scipy.cluster.hierarchy import ward
from sklearn.metrics.pairwise import cosine_similarity


def metaclusterdata_from_files(cluster_filepath, pixelcount_filepath, metacluster_header='metacluster'):  # noqa
    clusters = pd.read_csv(cluster_filepath)
    assert 'cluster' in clusters.columns, "cluster csv must include column named \"cluster\""
    assert metacluster_header in clusters.columns, "cluster csv must include column named \"metacluster\", alternately specify the metacluster indexs using keyword `metacluster_index`"  # noqa
    clusters = clusters.rename(columns={metacluster_header: 'metacluster'})
    pixelcounts = pd.read_csv(pixelcount_filepath)
    return MetaClusterData(clusters, pixelcounts)


class MetaClusterData():
    def __init__(self, raw_clusters_df, raw_pixelcounts_df):
        self.cluster_pixelcounts = raw_pixelcounts_df.sort_values('cluster').set_index('cluster')

        sorted_clusters_df = raw_clusters_df.sort_values('cluster')
        self._clusters = sorted_clusters_df.set_index('cluster').drop(columns='metacluster')
        self.mapping = sorted_clusters_df[['cluster', 'metacluster']].set_index('cluster')
        self._metacluster_displaynames_map = {}
        self._marker_order = list(range(len(self._clusters.columns)))

        assert len(set(self.clusters.index)) ==  len(list(self.clusters.index)), "Cluster ids must be unique."  # noqa
        assert set(self.clusters.index) == set(self.cluster_pixelcounts.index), "Cluster ids in both files must match"  # noqa
        assert 1 in self.clusters.index and 0 not in self.clusters.index, "Cluster ids must be integer, starting with 1."  # noqa

        self._output_mapping_filename = None
        self._cached_metaclusters = None

    @property
    def output_mapping_filename(self):
        return self._output_mapping_filename

    @output_mapping_filename.setter
    def output_mapping_filename(self, filepath):
        self._output_mapping_filename = Path(filepath)

    @property
    def clusters_with_metaclusters(self):
        df = self._clusters.join(self.mapping).sort_values(by='metacluster')
        return df.iloc[:, self._marker_order + [max(self._marker_order)+1]]

    @property
    def clusters(self):
        return self.clusters_with_metaclusters.drop(columns='metacluster')

    @property
    def metacluster_displaynames(self):
        return [self.get_metacluster_displayname(mc) for mc in self.metaclusters.index]

    @property
    def metaclusters(self):
        if self._cached_metaclusters is not None:
            return self._cached_metaclusters
        weighted_clusters = self.clusters.multiply(self.cluster_pixelcounts['count'], axis=0)
        metacluster_pixelcounts = self.cluster_pixelcounts.join(self.mapping) \
            .groupby('metacluster').aggregate('sum')
        weighted_metaclusters = weighted_clusters.join(self.mapping) \
            .groupby('metacluster').aggregate('sum') \
            .divide(metacluster_pixelcounts['count'], axis=0)
        self._cached_metaclusters = weighted_metaclusters
        return weighted_metaclusters

    @property
    def linkage_matrix(self):
        dist_matrix = cosine_similarity(self.clusters.T.values)
        linkage_matrix = ward(dist_matrix)
        return linkage_matrix

    def get_metacluster_displayname(self, metacluster):
        try:
            return self._metacluster_displaynames_map[metacluster]
        except KeyError:
            return str(metacluster)

    def cluster_in_metacluster(self, metacluster):
        return list(self.mapping[self.mapping['metacluster'] == metacluster].index.values)

    def which_metacluster(self, cluster):
        return self.mapping.loc[cluster]['metacluster']

    def new_metacluster(self):
        return max(self.mapping['metacluster']) + 1

    def remap(self, cluster, metacluster):
        self.mapping.loc[cluster, 'metacluster'] = metacluster
        self._cached_metaclusters = None

    def change_displayname(self, metacluster, displayname):
        self._metacluster_displaynames_map[metacluster] = displayname
        self.save_output_mapping()

    def save_output_mapping(self):
        out_df = self.mapping.copy()
        out_df['mc_name'] = [self.get_metacluster_displayname(mc) for mc in out_df['metacluster']]
        out_df.to_csv(self.output_mapping_filename)

    def set_marker_order(self, new_indexes):
        assert set(new_indexes) == set(self._marker_order), \
            f"New indexes ({new_indexes}) must be permuation of existing indexes ({self._marker_order})."  # noqa
        self._marker_order = new_indexes

    @property
    def cluster_count(self):
        return len(self.clusters)

    @property
    def metacluster_count(self):
        return len(set(self.mapping['metacluster']))

    @property
    def marker_count(self):
        return len(self.clusters.columns)

    @property
    def marker_names(self):
        return self.clusters.columns

    @property
    def fixed_width_marker_names(self):
        width = max(len(c) for c in self.marker_names)
        return [f"{c:^{width}}" for c in self.marker_names]
