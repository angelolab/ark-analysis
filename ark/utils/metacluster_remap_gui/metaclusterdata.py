from pathlib import Path

import pandas as pd
from scipy.stats import zscore


class MetaClusterData():
    def __init__(self, path, cluster_filename, pixelcount_filename, output_mapping_filename):
        self.path = Path(path)
        self.output_mapping_filename = output_mapping_filename
        clusters_raw = pd.read_csv(self.path / cluster_filename).sort_values('cluster')
        self.cluster_pixelcounts = pd.read_csv(self.path / pixelcount_filename) \
            .sort_values('cluster').set_index('cluster')
        self._clusters = clusters_raw.set_index('cluster').drop(columns='hCluster_cap')
        self.mapping = clusters_raw[['cluster', 'hCluster_cap']].set_index('cluster')
        self._cached_metaclusters = None

    @property
    def clusters_with_metaclusters(self):
        return self._clusters.join(self.mapping).sort_values(by='hCluster_cap')

    @property
    def clusters(self):
        return self.clusters_with_metaclusters.drop(columns='hCluster_cap')

    @property
    def metaclusters(self):
        if self._cached_metaclusters is not None:
            return self._cached_metaclusters
        weighted_clusters = self.clusters.multiply(self.cluster_pixelcounts['count'], axis=0)
        metacluster_pixelcounts = self.cluster_pixelcounts.join(self.mapping) \
            .groupby('hCluster_cap').aggregate('sum')
        weighted_metaclusters = weighted_clusters.join(self.mapping) \
            .groupby('hCluster_cap').aggregate('sum') \
            .divide(metacluster_pixelcounts['count'], axis=0)
        self._cached_metaclusters = weighted_metaclusters
        return weighted_metaclusters

    def cluster_in_metacluster(self, metacluster):
        return list(self.mapping[self.mapping['hCluster_cap'] == metacluster].index.values)

    def which_metacluster(self, cluster):
        return self.mapping.loc[cluster]['hCluster_cap']

    def new_metacluster(self):
        return max(self.mapping['hCluster_cap']) + 1

    def remap(self, cluster, metacluster):
        self.mapping.loc[cluster, 'hCluster_cap'] = metacluster
        self.mapping.to_csv(self.path / self.output_mapping_filename)
        self._cached_metaclusters = None

    @property
    def cluster_count(self):
        return len(self._clusters)

    @property
    def metacluster_count(self):
        return len(set(self.mapping['hCluster_cap']))

    @property
    def marker_count(self):
        return len(self.clusters.columns)

    @property
    def marker_names(self):
        return self.clusters.columns
