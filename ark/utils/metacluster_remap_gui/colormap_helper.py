import colorsys
import itertools

import matplotlib


def distinct_cmap(n=33):
    """Return a List of n visually distinct colors as a matplotlib ListedColorMap

    The sequence of color is deterministic for any n, and increasing n does not
    change the lower index colors.

    Args:
        n (int):
            The number of RGB tuples to return.
    Returns:
        matplotlib.colors.ListedColormap:
            N distinct colors as a matplotlib ListedColorMap
    """
    rgbs = distinct_rgbs(n)
    return matplotlib.colors.ListedColormap(rgbs)


def distinct_rgbs(n=33):
    """Return a List of n visually distinct colors as RGB tuples.

    The sequence of color is deterministic for any n, and increasing n does not
    change the lower index colors.

    Args:
        n (int):
            The number of RGB tuples to return.
    Returns:
        List[Tuple[int,int,int]]:
            List of the distinct colors as RGB tuples.
    """
    def infinite_hues():
        yield 0
        for k in itertools.count():
            i = 2 ** k  # zenos_dichotomy
            for j in range(1, i, 2):
                yield j / i

    def hue_to_hsvs(h):
        # tweak ratios to adjust scheme
        s = 6 / 10
        for v in [6 / 10, 9 / 10]:
            yield h, s, v

    hues = infinite_hues()
    hsvs = itertools.chain.from_iterable(hue_to_hsvs(hue) for hue in hues)
    rgbs = (colorsys.hsv_to_rgb(*hsv) for hsv in hsvs)
    return list(itertools.islice(rgbs, n))
