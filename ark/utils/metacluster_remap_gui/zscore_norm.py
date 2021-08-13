import numpy as np
from matplotlib.colors import Normalize


class ZScoreNormalize(Normalize):
    """Normalizer tailored for zscore heatmaps

    Creates two linear slops about zero using intervals:

        [min(array),0] and [0,max(array)]
    """
    def __call__(self, value, clip=None):
        """Map ndarray to the interval [0, 1]. The clip argument is unused."""
        result, is_scalar = self.process_value(value)
        assert not is_scalar, "This normalizer doesn't support scalars"

        self.vmin = np.min(result)
        self.vcenter = 0
        self.vmax = np.max(result)

        assert self.vmin <= self.vcenter <= self.vmax, \
            "vmin, vcenter, vmax must increase monotonically"

        result = np.ma.masked_array(
            np.interp(result, [self.vmin, self.vcenter, self.vmax],
                      [0, 0.5, 1.]), mask=np.ma.getmask(result))

        return result
