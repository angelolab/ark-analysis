
import pandas as pd
from alpineer import io_utils, misc_utils

from .metaclusterdata import MetaClusterData


def metaclusterdata_from_files(cluster_path, cluster_type='pixel', prefix_trim=None):
    """Read and validate raw CSVs and return an initialized MetaClusterData

    Args:
        cluster_path (str or IO):
            file path or filelike object
        cluster_type (str):
            the type of cluster data to read, needs to be either `'pixel'` or `'cell'`
        prefix_trim (str):
            If set, remove this prefix from each column of the data in `cluster_path`

    Returns:
        MetaClusterData:
            fully initialized metacluster data
    """

    # assert the path to the data is valid if a string
    if isinstance(cluster_path, str):
        io_utils.validate_paths(cluster_path)

    # assert the cluster type provided is valid
    misc_utils.verify_in_list(
        provided_cluster_type=[cluster_type],
        valid_cluster_types=['pixel', 'cell']
    )

    # read in the cluster data
    cluster_data = pd.read_csv(cluster_path)

    if prefix_trim is not None:
        cluster_data = cluster_data.rename(columns={
            col: col.replace(prefix_trim, '') for col in cluster_data.columns.values
        })

    # TODO: might want to rename and standardize everything in metacluster_remap_gui
    # with {cluster_type}_{som/meta}_cluster, not high priority
    cluster_data = cluster_data.rename(columns={
        '%s_som_cluster' % cluster_type: 'cluster',
        '%s_meta_cluster' % cluster_type: 'metacluster',
        '%s_meta_cluster_rename' % cluster_type: 'metacluster_rename'
    })

    if 'cluster' not in cluster_data.columns:
        raise ValueError("Cluster table must include column named \"cluster\"")

    if 'metacluster' not in cluster_data.columns:
        raise ValueError("Cluster table must include column named \"metacluster\"")

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

    return MetaClusterData(cluster_type, som_expression, som_counts)
