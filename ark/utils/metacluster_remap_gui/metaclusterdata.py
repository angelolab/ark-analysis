from pathlib import Path

import pandas as pd
from scipy.stats import zscore


def metaclusterdata_from_files(cluster_filepath, pixelcount_filepath, metacluster_header='hCluster_cap'):  # noqa
    clusters = pd.read_csv(cluster_filepath).rename(columns={metacluster_header: 'metacluster'})
    clusters = clusters.rename(columns={metacluster_header: 'metacluster'})
    pixelcounts = pd.read_csv(pixelcount_filepath)
    return MetaClusterData(clusters, pixelcounts)


class MetaClusterData():
    def __init__(self, raw_clusters_df, raw_pixelcounts_df):
        self.cluster_pixelcounts = raw_pixelcounts_df.sort_values('cluster').set_index('cluster')

        sorted_clusters_df = raw_clusters_df.sort_values('cluster')
        self._clusters = sorted_clusters_df.set_index('cluster').drop(columns='metacluster')
        self.mapping = sorted_clusters_df[['cluster', 'metacluster']].set_index('cluster')

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
        return self._clusters.join(self.mapping).sort_values(by='metacluster')

    @property
    def clusters(self):
        return self.clusters_with_metaclusters.drop(columns='metacluster')

    @property
    def metacluster_displaynames(self):
        return [str(mc) for mc in self.metaclusters.index]

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

    def cluster_in_metacluster(self, metacluster):
        return list(self.mapping[self.mapping['metacluster'] == metacluster].index.values)

    def which_metacluster(self, cluster):
        return self.mapping.loc[cluster]['metacluster']

    def new_metacluster(self):
        return max(self.mapping['metacluster']) + 1

    def remap(self, cluster, metacluster):
        self.mapping.loc[cluster, 'metacluster'] = metacluster
        self.save_output_mapping()
        self._cached_metaclusters = None

    def save_output_mapping(self):
        self.mapping.to_csv(self.output_mapping_filename)

    @property
    def cluster_count(self):
        return len(self._clusters)

    @property
    def metacluster_count(self):
        return len(set(self.mapping['metacluster']))

    @property
    def marker_count(self):
        return len(self.clusters.columns)

    @property
    def marker_names(self):
        return self.clusters.columns
