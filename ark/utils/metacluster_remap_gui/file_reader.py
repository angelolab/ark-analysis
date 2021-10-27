import pandas as pd

from .metaclusterdata import MetaClusterData


def metaclusterdata_from_files(cluster_io, pixelcount_io, metacluster_header='metacluster'):
    """Read and validate raw CSVs and return an initialized MetaClusterData

        Args:
            cluster_io (IO)):
                file path or filelike object
            pixelcount_io (IO)):
                file path or filelike object
            metacluster_header (str):
                alternate header which can contains the metacluster ids
        Returns:
            MetaClusterData:
                Fully initialized metacluster data
    """
    clusters = pd.read_csv(cluster_io)

    if 'cluster' not in clusters.columns:
        raise ValueError("cluster csv must include column named \"cluster\"")

    if metacluster_header not in clusters.columns:
        raise ValueError("cluster csv must include column named \"metacluster\", alternately specify the metacluster indexs using keyword `metacluster_index`")  # noqa

    clusters = clusters.rename(columns={metacluster_header: 'metacluster'})
    pixelcounts = pd.read_csv(pixelcount_io)

    if 'cluster' not in pixelcounts.columns:
        raise ValueError("pixelcounts csv must include column named \"cluster\"")

    if 'count' not in pixelcounts.columns:
        raise ValueError("pixelcounts csv must include column named \"count\"")

    if len(set(clusters['cluster'].values)) != len(list(clusters['cluster'].values)):
        raise ValueError("Cluster ids must be unique.")

    if 1 not in clusters['cluster'].values:
        raise ValueError("Cluster ids must be int type, starting with 1.")

    if 0 in clusters['cluster'].values:
        raise ValueError("Cluster ids start with 1, but a zero was detected.")

    if set(clusters['cluster'].values) != set(pixelcounts['cluster'].values):
        raise ValueError("Cluster ids in both files must match")

    return MetaClusterData(clusters, pixelcounts)
