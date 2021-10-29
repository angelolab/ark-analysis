from .colormap_helper import distinct_cmap, distinct_rgbs


def test_colormap_is_distinct():
    assert len(set(distinct_rgbs(200))) == 200


def test_colormap_runs():
    distinct_cmap(10)
