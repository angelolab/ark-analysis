# Composite builder functionality
# combine individual channels or pixel cluster tiffs into a single composite channel

import numpy as np

# Function: composite_builder
# Adds tiffs together, either pixel clustered or base signal tiffs and returns a composite channel or mask.
#
# images_to_add = a list of channels or pixel cluster names to add together
# image_to_subtract = a list of channels ot pixel cluster names to subtract from the composite
# image_type = intensity or pixel cluster data
# composite_method = binarized mask returns ('full') or intensity, gray-scale tiffs returned ('partial')

# returns the composite array, either in binarized mask or scaled intensity format.

def composite_builder(images_to_add=[], images_to_subtract=[], image_type='intensity', composite_method='partial'):
    # set up composite array dimensions using images to be added or if none, images to be subtracted.
    try:
        image_dim = images_to_add[0].shape
    except IndexError:
        image_dim = images_to_subtract[0].shape
    composite_array = np.zeros(shape=image_dim)
    if images_to_add:
        composite_array = add_to_composite(composite_array, images_to_add, image_type, composite_method)
    if images_to_subtract:
        composite_array = subtract_from_composite(images_to_subtract, image_type, composite_method)
    return composite_array


# Function: add_to_composite
# Adds tiffs together to form a composite array
#
# composite_array = an array to add tiffs to
# images_to_add = a list of channels to pixel cluster names to subtract from the composite
# image_type = intensity or pixel cluster data
# composite_method = binarized mask returns ('full') or intensity, gray-scale tiffs returned ('partial')

# returns the composite array, either in binarized mask or scaled intensity format.

def add_to_composite(composite_array, images_to_add, image_type, composite_method):
    # for each channel to add
    for channel in images_to_add:
        # if intensity data
        if image_type == 'intensity':
            # add channel counts into composite array
            composite_array = np.add(composite_array, channel)
            # if a binarized, or full removal is asked for
            if composite_method == 'full':
                composite_array[composite_array > 1] = 1
        # else if pixel clustered data
        elif image_type == 'pixel_clustered':
            # add positive pixels to composite array
            composite_array = np.bitwise_or(composite_array, channel)
    # return the composite array
    return composite_array

# Function: subtract_from_composite
# Subtract tiffs from a composite array
#
# # composite_array = an array to subtract tiffs from
# image_to_subtract = a list of channels to pixel cluster names to subtract from the composite
# image_type = intensity or pixel cluster data
# composite_method = binarized mask returns ('full') or intensity, gray-scale tiffs returned ('partial')

# returns the composite array, either in binarized mask or scaled intensity format.

def subtract_from_composite(composite_array, images_to_subtract, image_type, composite_method):
    # for each channel to subtract
    for channel in images_to_subtract:
        # if intensity data
        if image_type == 'intensity':
            # if a binarized, or full removal is asked for
            if composite_method == 'full':
                # Create a mask based on positive values in the subtraction channel
                mask_2_zero = channel > 0
                # Zero out elements in the composite channel based on mask
                composite_array[mask_2_zero] = 0
                # return binarized composite
                composite_array[composite_array > 1] = 1
            # if a signal based, or partial, removal is asked for
            elif composite_method == 'partial':
                # subtract channel counts from composite array
                composite_array = np.subtract(composite_array, channel)
                # Find the minimum value in the composite array
                min_value = composite_array.min()
                # If the minimum value is negative, add its absolute value to all elements
                if min_value < 0:
                    composite_array += abs(min_value)
        # else if pixel clustered data
        elif image_type == 'pixel_clustered':
            # subtract positive pixels to composite array
            composite_array = np.subtract(composite_array, channel)
            # zero out any negative values
            composite_array[composite_array < 0] = 0
    # return the composite array
    return composite_array
