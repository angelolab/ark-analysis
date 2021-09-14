import numpy as np
from matplotlib.colors import Normalize


class ZScoreNormalize(Normalize):
    """Normalizer tailored for zscore heatmaps

    Creates two linear slops about zero using intervals:

        [min(array),0] and [0,max(array)]
    """
    def __init__(self, vmin=-1, vcenter=0, vmax=1):
        self.vcenter = vcenter
        super().__init__(vmin, vmax)

    def calibrate(self, values):
        self.vmin = np.min(values)
        if (self.vmin > 0):
            self.vmin = 0.0
        self.vcenter = 0.0
        self.vmax = np.max(values)

        assert self.vmin <= self.vcenter <= self.vmax, \
            f"vmin({self.vmin:0.0f}), vcenter({self.vcenter:0.0f}), vmax({self.vmax:0.0f}) must increase monotonically"  # noqa

    def __call__(self, value, clip=None):
        """Map ndarray to the interval [0, 1]. The clip argument is unused."""
        result, is_scalar = self.process_value(value)
        assert not is_scalar, "This normalizer doesn't support scalars"

        normalized_values = np.interp(
            result,
            [self.vmin, self.vcenter, self.vmax],
            [0, 0.5, 1.])

        return np.ma.masked_array(normalized_values, mask=np.ma.getmask(result))
