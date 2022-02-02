import os
import pandas as pd

from ark.utils import misc_utils
from .metaclusterdata import MetaClusterData


def metaclusterdata_from_files_old(cluster_io, pixelcount_io, metacluster_header='metacluster'):
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


def metaclusterdata_from_files(cluster_io, cluster_type='pixel'):
    """Read and validate raw CSVs and return an initialized MetaClusterData

    Args:
        cluster_io (IO):
            file path or filelike object
        som_cluster_header (str):
            the name of the SOM cluster, must be `'pixel_som_cluster'` or `'cell_som_cluster'`
        meta_cluster_header (str):
            the name of the meta cluster, must be `'pixel_meta_cluster'` or `'cell_som_cluster'`

    Returns:
        MetaClusterData:
            fully initialized metacluster data
    """

    # assert the path to the data is valid
    if not os.path.exists(cluster_io):
        raise FileNotFoundError('Path to clustering data %s does not exist' % cluster_io)

    # assert the cluster type provided is valid
    misc_utils.verify_in_list(
        provided_cluster_type=[cluster_type],
        valid_cluster_types=['pixel', 'cell']
    )

    # read in the cluster data
    cluster_data = pd.read_csv(cluster_io)

    # TODO: we should remove this rename and standardize everything in metacluster_remap_gui
    # with {cluster_type}_{som/meta}_cluster
    cluster_data = cluster_data.rename(columns={
        '%s_som_cluster' % cluster_type: 'cluster',
        '%s_meta_cluster' % cluster_type: 'metacluster'
    })

    if 'cluster' not in cluster_data.columns:
        raise ValueError("Cluster table must include column named \"cluster\"")

    if 'count' not in cluster_data.columns:
        raise ValueError("Cluster table must include column named \"count\"")

    if len(set(cluster_data['cluster'].values)) != len(list(cluster_data['cluster'].values)):
        raise ValueError("SOM cluster ids must be unique")

    if 1 not in cluster_data['cluster'].values:
        raise ValueError("SOM cluster ids must be int type, starting with 1.")

    if 0 in cluster_data['cluster'].values:
        raise ValueError("SOM cluster ids start with 1, but a zero was detected.")

    # extract the SOM cluster counts separately
    som_counts = cluster_data[['cluster', 'count']].copy()

    # drop the 'count' column from the cluster_data to produce the averages table
    # NOTE: channel avg for pixel clusters, pixel count avg for cell clusters
    som_expression = cluster_data.drop(columns='count')

    return MetaClusterData(som_expression, som_counts)
