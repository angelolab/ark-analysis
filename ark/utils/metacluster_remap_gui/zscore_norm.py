import numpy as np
from matplotlib.colors import Normalize


class ZScoreNormalize(Normalize):
    """Normalizer tailored for zscore heatmaps

    Map each value of an incoming vector each between 0 and 1, which
    is the interval for cmaps.

    The mapping consists of two separate linearly interpolated intervals:

        [vmin,vcenter] -> [0.0,0.5]
        [vcenter,vmax] -> [0.5,1.0]

    """
    def __init__(self, vmin=-3, vcenter=0, vmax=3):
        """Initial ZScoreNormalize

        vmin < vcenter < vmax

        Args:
            vmin (float):
                Value to map to 0 in the colormap
            vcenter (float):
                Value to map to .5 in the colormap
            vmax (float):
                Value to map to 1 in the colormap
        """
        self.vcenter = vcenter
        super().__init__(vmin, vmax)

    def inverse(self, value):
        result = np.interp(
            value,
            [0, 0.5, 1],
            [self.vmin, self.vcenter, self.vmax],
        )
        return result

    def calibrate(self, values):
        self.vmin = min([-np.max(values), 0])
        self.vcenter = 0.0
        self.vmax = np.max(values)

    def __call__(self, value: np.ndarray, clip=None):
        """Map ndarray to the interval [0, 1]. The clip argument is unused."""
        result, is_scalar = self.process_value(value)

        normalized_values = np.interp(
            result,
            [self.vmin, self.vcenter, self.vmax],
            [0, 0.5, 1.]
        )

        return np.ma.masked_array(normalized_values, mask=np.ma.getmask(result))
