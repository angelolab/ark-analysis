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
    mcg.selected_clusters = {1, 2}
    mcg.remap_current_selection(3)
    assert mcg.mcd.which_metacluster(1) == 3
    assert mcg.mcd.which_metacluster(2) == 3


def test_enable_debug_mode(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata, debug=True)


def test_update_zscore(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.zscore_clamp_slider.value += 1


def test_new_metacluster(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.selected_clusters.add(1)
    mcg.new_metacluster(None)
    assert simple_metaclusterdata.mapping.loc[1, 'metacluster'] == 4


class DummyClick:
    def __init__(self, artist, x, y=None, is_rightclick=False):
        self.artist = artist

        class MouseEvent:
            pass
        self.mouseevent = MouseEvent()
        self.mouseevent.name = 'button_press_event'
        self.mouseevent.xdata = x
        self.mouseevent.ydata = y
        self.mouseevent.button = 3 if is_rightclick else 1


def test_can_select_cluster_in_cluster_heatmap(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    dummyclick = DummyClick(mcg.im_c, 0.5)
    mcg.onpick(dummyclick)
    assert mcg.selected_clusters == {1}


def test_can_deselect_clusters_in_cluster_heatmap(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.selected_clusters = {1, 2, 3, 4}
    dummyclick = DummyClick(mcg.im_c, 0.5)
    mcg.onpick(dummyclick)
    assert mcg.selected_clusters == {2, 3, 4}


def test_can_pick_metacluster_in_metacluster_heatmap(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    dummyclick = DummyClick(mcg.im_m, 2.5)
    mcg.onpick(dummyclick)
    assert mcg.selected_clusters == {3, 4}


def test_can_select_metacluster_color_labels(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    dummyclick = DummyClick(mcg.im_cl, 3.5)
    mcg.onpick(dummyclick)
    assert mcg.selected_clusters == {3, 4}


def test_can_deselect_metacluster_color_labels(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.selected_clusters = {1, 2, 3, 4}
    dummyclick = DummyClick(mcg.im_cl, 3.5)
    mcg.onpick(dummyclick)
    assert mcg.selected_clusters == {1, 2}


def test_can_remap_by_cluster(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.selected_clusters = {1}
    dummyclick = DummyClick(mcg.im_c, 3.5, is_rightclick=True)
    mcg.onpick(dummyclick)
    assert mcg.mcd.which_metacluster(1) == 3


def test_can_remap_by_cluster_color_label(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.selected_clusters = {1}
    dummyclick = DummyClick(mcg.im_cl, 3.5, is_rightclick=True)
    mcg.onpick(dummyclick)
    assert mcg.mcd.which_metacluster(1) == 3


def test_can_remap_by_metacluster(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata)
    mcg.selected_clusters = {1}
    dummyclick = DummyClick(mcg.im_m, 2.5, is_rightclick=True)
    mcg.onpick(dummyclick)
    assert mcg.mcd.which_metacluster(1) == 3
