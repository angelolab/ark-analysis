from pathlib import Path

from .metaclusterdata import MetaClusterData
from .metaclustergui import MetaClusterGui
from .test_metaclusterdata import simple_metaclusterdata

THIS_DIR = Path(__file__).parent


def test_can_create_metaclustergui(simple_metaclusterdata: MetaClusterData):
    MetaClusterGui(simple_metaclusterdata)


def test_can_select_cluster(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.selected_clusters.add(2)
    assert mcg.selected_clusters == set([2])


def test_can_select_all_clusters_in_metacluster(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.select_metacluster(3)
    assert len(mcg.selected_clusters) == 2


def test_can_clear_selection(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.selected_clusters.add(2)
    mcg.clear_selection(None)
    assert len(mcg.selected_clusters) == 0


def test_can_remap_all_selected(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.selected_clusters.add(1)
    mcg.selected_clusters.add(2)
    mcg.remap_current_selection(3)
    assert simple_metaclusterdata.mapping.loc[1, 'metacluster'] == 3
    assert simple_metaclusterdata.mapping.loc[2, 'metacluster'] == 3


def test_can_pick_cell_in_heatmap(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.ax_c.pick


def test_enable_debug_mode(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata, debug=True)


def test_update_zscore(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata, debug=True)
    mcg.zscore_clamp_slider.value += 1
