import colorsys
import itertools

import matplotlib


def distinct_cmap(n=33):
    def infinite_hues():
        yield 0
        for k in itertools.count():
            i = 2 ** k  # zenos_dichotomy
            for j in range(1, i, 2):
                yield j / i

    def hue_to_hsvs(h):
        # tweak ratios to adjust scheme
        s = 6/10
        for v in [6/10, 9/10]:
            yield h, s, v

    hues = infinite_hues()
    hsvs = itertools.chain.from_iterable(hue_to_hsvs(hue) for hue in hues)
    rgbs = (colorsys.hsv_to_rgb(*hsv) for hsv in hsvs)
    return matplotlib.colors.ListedColormap(list(itertools.islice(rgbs, n)))
