"""This file contains all filters (kernels), i.e. icky-yucky image processing stuff."""

import numpy as np
import vigra

def prepare_data(data, slice):
    return data[slice]

def gaussian(sigma, filter_size):
    pass


def laplacian(sigma):
    pass

def get_kernel(kernel_name, *args, **kwargs):
    """General interface to fetch 2D kernels."""
    _ALL_KERNELS = {'gaussian': gaussian,
                    'laplacian': laplacian}
    _nD_kernel = _ALL_KERNELS.get(kernel_name.to_lower())(*args, **kwargs)
    # Torch expects a (n+1)D kernel, where the leading dimension corresponds to output channel,
    # and the trailing dimensions are spatial.
    _np1D_kernel = _nD_kernel[None, ...]
    return _np1D_kernel
