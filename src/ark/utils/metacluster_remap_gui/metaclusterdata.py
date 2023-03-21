from pathlib import Path

from scipy.cluster.hierarchy import ward
from sklearn.metrics.pairwise import cosine_similarity


class MetaClusterData():
    """Store the state of the clusters and metaclusters

    Args:
        cluster_type (str):
            the type of clustering being done
        raw_clusters_df (pd.Dataframe):
            validated and initialized clusters dataframe.
        raw_pixelcounts_df (pd.Dataframe):
            validated and initialized pixelcounts dataframe.
    """
    def __init__(self, cluster_type, raw_clusters_df, raw_pixelcounts_df):
        self.cluster_type = cluster_type
        self.cluster_pixelcounts = raw_pixelcounts_df.sort_values('cluster').set_index('cluster')

        sorted_clusters_df = raw_clusters_df.sort_values('cluster')
        self._clusters = sorted_clusters_df.set_index('cluster').drop(columns='metacluster')
        self.mapping = sorted_clusters_df[['cluster', 'metacluster']].set_index('cluster')
        self._metacluster_displaynames_map = {}

        # need to prefill the displaynames_map with the already renamed meta clusters
        # on subsequent runs after the first to prevent automatic incremental rewriting
        if 'metacluster_rename' in sorted_clusters_df.columns:
            unique_mappings = sorted_clusters_df[
                ['metacluster', 'metacluster_rename']
            ].drop_duplicates()

            self._metacluster_displaynames_map = {
                mc['metacluster']: str(mc['metacluster_rename'])
                for _, mc in unique_mappings.iterrows()
            }

        self._marker_order = list(range(len(self._clusters.columns)))
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

        # NOTE: this method takes into account both the initial run (without _rename column)
        # and subsequent runs (with _rename columns)
        return df.iloc[:, self._marker_order + list(
            range(max(self._marker_order) + 1, len(df.columns.values))
        )]

    @property
    def clusters(self):
        # maintain old clusters_with_metaclusters
        clusters_data = self.clusters_with_metaclusters.copy()

        # we need to drop the rename column on subsequent runs after the first
        if 'metacluster_rename' in self.clusters_with_metaclusters.columns:
            clusters_data = clusters_data.drop(columns='metacluster_rename')

        # metacluster column needs to be dropped regardless of run
        return clusters_data.drop(columns='metacluster')

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
        out_df.index.names = [f'{self.cluster_type}_som_cluster']
        out_df[f'{self.cluster_type}_meta_cluster_rename'] = [
            self.get_metacluster_displayname(mc) for mc in out_df['metacluster']
        ]
        out_df = out_df.rename(columns={'metacluster': f'{self.cluster_type}_meta_cluster'})
        out_df.to_csv(self.output_mapping_filename)

    def set_marker_order(self, new_indexes):
        self._marker_order = new_indexes
        self._cached_metaclusters = None

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
