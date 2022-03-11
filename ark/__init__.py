# This should ignore the low contrast warning from skimage. skimage uses imageio under the hood 
# to write to files.
# Reference https://stackoverflow.com/questions/52486237/how-to-turn-off-skimage-warnings
import imageio.core.util

def ignore_warnings(*args, **kwargs):
    """Ignores the low contrast warning
    """
    pass

imageio.core.util._precision_warn = ignore_warnings