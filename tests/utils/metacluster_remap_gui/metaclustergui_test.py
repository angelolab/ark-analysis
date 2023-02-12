import asyncio
from pathlib import Path

import pytest

from ark.utils.metacluster_remap_gui.metaclusterdata import MetaClusterData
from ark.utils.metacluster_remap_gui.metaclustergui import MetaClusterGui

THIS_DIR = Path(__file__).parent

pytestmark = pytest.mark.filterwarnings("ignore:coroutine*:RuntimeWarning")


@pytest.fixture
def mcg(simple_metaclusterdata: MetaClusterData):
    return MetaClusterGui(simple_metaclusterdata, enable_throttle=False)


@pytest.fixture(autouse=True, scope='session')
def use_pseudo_inverse():
    import matplotlib.transforms as f
    from numpy.linalg import inv, pinv
    f.inv = pinv
    yield
    f.inv = inv


def test_can_create_metaclustergui(mcg: MetaClusterGui):
    mcg


def test_enable_debug_mode(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata, debug=True)


@pytest.mark.asyncio
async def test_can_run_asyncio_pieces_of_gui_refresh(simple_metaclusterdata: MetaClusterData):
    mcg = MetaClusterGui(simple_metaclusterdata, enable_throttle=True)
    mcg._heatmaps_stale = True

    mcg.update_gui()
    assert mcg._heatmaps_stale

    while mcg._heatmaps_stale:
        await asyncio.sleep(0.01)
    assert not mcg._heatmaps_stale

    # and run shortcut update as well
    mcg.update_gui()
    assert not mcg._heatmaps_stale

    # let it run
    await asyncio.sleep(0.4)


def test_can_select_cluster(mcg: MetaClusterGui):
    mcg.selected_clusters.add(2)
    assert mcg.selected_clusters == set([2])


def test_can_select_all_clusters_in_metacluster(mcg: MetaClusterGui):
    mcg.select_metacluster(3)
    assert len(mcg.selected_clusters) == 2


def test_can_clear_selection(mcg: MetaClusterGui):
    mcg.selected_clusters.add(2)
    mcg.clear_selection(None)
    assert len(mcg.selected_clusters) == 0


def test_can_remap_all_selected(mcg: MetaClusterGui):
    mcg.selected_clusters = {1, 2}
    mcg.remap_current_selection(3)
    assert mcg.mcd.which_metacluster(1) == 3
    assert mcg.mcd.which_metacluster(2) == 3


def test_update_zscore(mcg: MetaClusterGui):
    mcg.zscore_clamp_slider.value += 1


def test_update_zscore_fractional(mcg: MetaClusterGui):
    mcg.zscore_clamp_slider.value += 0.5


def test_new_metacluster(mcg: MetaClusterGui):
    mcg.selected_clusters.add(1)
    mcg.new_metacluster(None)
    assert mcg.mcd.mapping.loc[1, 'metacluster'] == 4


class DummyClick:
    def __init__(self, artist, x, y=None, is_rightclick=False, event_type='button_press_event'):
        self.artist = artist

        class MouseEvent:
            pass
        self.mouseevent = MouseEvent()
        self.mouseevent.name = event_type
        self.mouseevent.xdata = x
        self.mouseevent.ydata = y
        self.mouseevent.button = 3 if is_rightclick else 1


def test_handler_ignore_non_clicks(mcg: MetaClusterGui):
    dummyclick = DummyClick(mcg.im_c, 0.5, event_type='fake')
    mcg.onpick(dummyclick)
    assert mcg.selected_clusters == set()


def test_can_select_cluster_in_cluster_heatmap(mcg: MetaClusterGui):
    dummyclick = DummyClick(mcg.im_c, 0.5)
    mcg.onpick(dummyclick)
    assert mcg.selected_clusters == {1}


def test_can_deselect_clusters_in_cluster_heatmap(mcg: MetaClusterGui):
    mcg.selected_clusters = {1, 2, 3, 4}
    dummyclick = DummyClick(mcg.im_c, 0.5)
    mcg.onpick(dummyclick)
    assert mcg.selected_clusters == {2, 3, 4}


def test_can_pick_metacluster_in_metacluster_heatmap(mcg: MetaClusterGui):
    dummyclick = DummyClick(mcg.im_m, 2.5)
    mcg.onpick(dummyclick)
    assert mcg.selected_clusters == {3, 4}


def test_can_select_metacluster_color_labels(mcg: MetaClusterGui):
    dummyclick = DummyClick(mcg.im_cl, 3.5)
    mcg.onpick(dummyclick)
    assert mcg.selected_clusters == {3, 4}


def test_can_deselect_metacluster_color_labels(mcg: MetaClusterGui):
    mcg.selected_clusters = {1, 2, 3, 4}
    dummyclick = DummyClick(mcg.im_cl, 3.5)
    mcg.onpick(dummyclick)
    assert mcg.selected_clusters == {1, 2}


def test_can_remap_by_cluster(mcg: MetaClusterGui):
    mcg.selected_clusters = {1}
    dummyclick = DummyClick(mcg.im_c, 3.5, is_rightclick=True)
    mcg.onpick(dummyclick)
    assert mcg.mcd.which_metacluster(1) == 3


def test_can_remap_by_cluster_color_label(mcg: MetaClusterGui):
    mcg.selected_clusters = {1}
    dummyclick = DummyClick(mcg.im_cl, 3.5, is_rightclick=True)
    mcg.onpick(dummyclick)
    assert mcg.mcd.which_metacluster(1) == 3


def test_can_remap_by_metacluster(mcg: MetaClusterGui):
    mcg.selected_clusters = {1}
    dummyclick = DummyClick(mcg.im_m, 2.5, is_rightclick=True)
    mcg.onpick(dummyclick)
    assert mcg.mcd.which_metacluster(1) == 3


def test_selection_mask(mcg: MetaClusterGui):
    mcg.selected_clusters.add(2)
    assert mcg.selection_mask == [[0, 1, 0, 0]]
