import numpy as np

from ark.utils.metacluster_remap_gui.metaclusterdata import MetaClusterData


def test_can_get_mapping(simple_metaclusterdata: MetaClusterData,
                         simple_metaclusterdata_rename: MetaClusterData):
    np.testing.assert_array_equal(
        simple_metaclusterdata.mapping['metacluster'].values,
        np.array((1, 2, 3, 3))
    )

    np.testing.assert_array_equal(
        simple_metaclusterdata_rename.mapping['metacluster'].values,
        np.array((1, 2, 3, 3))
    )


def test_can_remap(simple_metaclusterdata: MetaClusterData,
                   simple_metaclusterdata_rename: MetaClusterData):
    simple_metaclusterdata.remap(4, 1)
    assert simple_metaclusterdata.mapping.loc[4, 'metacluster'] == 1

    simple_metaclusterdata_rename.remap(4, 1)
    assert simple_metaclusterdata_rename.mapping.loc[4, 'metacluster'] == 1


def test_can_create_new_metacluster(simple_metaclusterdata: MetaClusterData,
                                    simple_metaclusterdata_rename: MetaClusterData):
    new_mc = simple_metaclusterdata.new_metacluster()

    simple_metaclusterdata.remap(4, new_mc)
    assert simple_metaclusterdata.mapping.loc[4, 'metacluster'] == 4

    simple_metaclusterdata_rename.remap(4, new_mc)
    assert simple_metaclusterdata_rename.mapping.loc[4, 'metacluster'] == 4


def test_can_save_mapping(simple_metaclusterdata: MetaClusterData,
                          simple_metaclusterdata_rename: MetaClusterData,
                          tmp_path):
    simple_metaclusterdata.output_mapping_filename = tmp_path / 'output_mapping.csv'
    simple_metaclusterdata.save_output_mapping()
    with open(tmp_path / 'output_mapping.csv', 'r') as f:
        output = [ll.strip() for ll in f.readlines()]
    assert output == [
        "pixel_som_cluster,pixel_meta_cluster,pixel_meta_cluster_rename",
        "1,1,1",
        "2,2,2",
        "3,3,3",
        "4,3,3",
    ]

    simple_metaclusterdata_rename.output_mapping_filename = tmp_path / 'output_mapping.csv'
    simple_metaclusterdata_rename.save_output_mapping()
    with open(tmp_path / 'output_mapping.csv', 'r') as f:
        output = [ll.strip() for ll in f.readlines()]
    assert output == [
        "cell_som_cluster,cell_meta_cluster,cell_meta_cluster_rename",
        "1,1,cluster_1",
        "2,2,cluster_2",
        "3,3,cluster_3",
        "4,3,cluster_3",
    ]


def test_metaclusters_can_have_displaynames(simple_metaclusterdata: MetaClusterData,
                                            simple_metaclusterdata_rename: MetaClusterData):
    assert simple_metaclusterdata.metacluster_displaynames == ['1', '2', '3']
    assert simple_metaclusterdata_rename.metacluster_displaynames == \
        ['cluster_1', 'cluster_2', 'cluster_3']


def test_metaclusters_can_change_displaynames(simple_metaclusterdata: MetaClusterData,
                                              simple_metaclusterdata_rename: MetaClusterData):
    simple_metaclusterdata.change_displayname(1, 'y2k')
    assert simple_metaclusterdata.metacluster_displaynames == ['y2k', '2', '3']

    simple_metaclusterdata_rename.change_displayname(1, 'y2k')
    assert simple_metaclusterdata_rename.metacluster_displaynames == \
        ['y2k', 'cluster_2', 'cluster_3']


def test_can_match_cluster_to_metacluster(simple_metaclusterdata: MetaClusterData,
                                          simple_metaclusterdata_rename: MetaClusterData):
    assert simple_metaclusterdata.which_metacluster(4) == 3

    assert simple_metaclusterdata_rename.which_metacluster(4) == 3


def test_can_average_clusters_by_metacluster(simple_metaclusterdata: MetaClusterData,
                                             simple_metaclusterdata_rename: MetaClusterData):
    simple_metaclusterdata.remap(4, 3)
    clusters_data = np.array([
        (0.1, 0.2, 0.1),
        (0.1, 0.1, 0.3),
        ((0.5 * 50 + 0.7 * 77) / (50 + 77),
         (0.1 * 50 + 0.2 * 77) / (50 + 77),
         (0.1 * 50 + 0.1 * 77) / (50 + 77)),
    ])
    np.testing.assert_equal(simple_metaclusterdata.metaclusters.values, clusters_data)

    simple_metaclusterdata_rename.remap(4, 3)
    clusters_data = np.array([
        (0.1, 0.2, 0.1),
        (0.1, 0.1, 0.3),
        ((0.5 * 50 + 0.7 * 77) / (50 + 77),
         (0.1 * 50 + 0.2 * 77) / (50 + 77),
         (0.1 * 50 + 0.1 * 77) / (50 + 77)),
    ])
    np.testing.assert_equal(simple_metaclusterdata_rename.metaclusters.values, clusters_data)


def test_can_reorder_markers(simple_metaclusterdata: MetaClusterData,
                             simple_metaclusterdata_rename: MetaClusterData):
    simple_metaclusterdata.set_marker_order([0, 2, 1])
    assert list(simple_metaclusterdata.marker_names) == ['CD163', 'CD31', 'CD206']

    simple_metaclusterdata_rename.set_marker_order([0, 2, 1])
    assert list(simple_metaclusterdata_rename.marker_names) == ['CD163', 'CD31', 'CD206']


def test_marker_orders_match(simple_metaclusterdata: MetaClusterData,
                             simple_metaclusterdata_rename: MetaClusterData):
    # access the properties first to reproduce a cache invalidation bug
    _ = simple_metaclusterdata.clusters
    _ = simple_metaclusterdata.metaclusters
    _ = simple_metaclusterdata.clusters_with_metaclusters
    simple_metaclusterdata.set_marker_order([0, 2, 1])
    c_marks = list(simple_metaclusterdata.clusters.columns[0:3])
    m_marks = list(simple_metaclusterdata.metaclusters.columns[0:3])
    assert c_marks == m_marks

    # access the properties first to reproduce a cache invalidation bug
    _ = simple_metaclusterdata_rename.clusters
    _ = simple_metaclusterdata_rename.metaclusters
    _ = simple_metaclusterdata_rename.clusters_with_metaclusters
    simple_metaclusterdata_rename.set_marker_order([0, 2, 1])
    c_marks = list(simple_metaclusterdata_rename.clusters.columns[0:3])
    m_marks = list(simple_metaclusterdata_rename.metaclusters.columns[0:3])
    assert c_marks == m_marks
