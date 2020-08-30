from contextlib import contextmanager
from argparse import Namespace

import time
import h5py
import numpy as np
from itertools import tee
# from torchy.filters.vigra_filters import get_filter_size


@contextmanager
def timeit():
    """
    Context manager that times what happens under it. Maybe with a wee-bit overhead,
    but shouldn't matter much for relative measurements.

    Usage:
    ```
    with timeit() as timeit_info:
        output = ...

    print("Elapsed time: {}".format(timit_info.elapsed_time))
    ```
    """
    timeit_info = Namespace()
    now = time.time()
    yield timeit_info
    later = time.time()
    timeit_info.elapsed_time = later - now


def simulate_delay(duration):
    """Delay a given function by a certain `duration` (in seconds)."""
    def _decorator(function):
        def _function(*args, **kwargs):
            time.sleep(duration)
            return function(*args, **kwargs)
        return _function
    return _decorator


@contextmanager
def no_lock():
    yield


def load_raw_data(filename, slice_with_halo):
    """
    wrapper for h5py that slices the ROI and reshapes to the 
    format (batch, channel, x, y, z).
    For raw input data batch = channel =1
    """
    with h5py.File(filename, "r") as f5:
        dset = f5['volume/data']
        out = np.array(dset[slice_with_halo])
    new_shape = [1, 1] + list(out.shape)
    return out.reshape(new_shape)


def get_filter_size(filter_name):
    if filter_name == 'Gaussian Smoothing':
        return 1
    elif filter_name == 'Laplacian of Gaussian':
        return 1
    elif filter_name == 'Hessian of Gaussian Eigenvalues':
        return 3
    elif filter_name == 'Gaussian Gradient Magnitude':
        return 1
    else:
        raise NotImplementedError


def get_feature_index(r):
    """
    function that returns the slices that generate the feature file in 
    matching request r
    """

    feature_names_to_index = {'Gaussian Smoothing': [0],
                              'Gaussian Gradient Magnitude': [1],
                              'Laplacian of Gaussian': [2],
                              'Hessian of Gaussian Eigenvalues': [3, 4]}

    sigma_to_index = {0.3: [0],
                      0.7: [1],
                      1.0: [2],
                      1.6: [3],
                      3.5: [4],
                      5.0: [5],
                      10.0: [6]}

    s_list = []
    f_list = []

    for f in r['features']:
        feature_index = feature_names_to_index[f['name']]
        s_list.extend(sigma_to_index[f['sigma']] * len(feature_index))
        f_list.extend(feature_index)

    return s_list, f_list


def reshape_volume_for_torch(volume):
    assert volume.ndim == 3 or volume.ndim == 2
    return volume[None, None, ...]

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


if __name__ == '__main__':
    print(get_feature_index({'features': [{'name': 'Hessian of Gaussian Eigenvalues', 'sigma': 1.6},
                                          {'name': 'Hessian of Gaussian Eigenvalues', 'sigma': 10.}]}))
