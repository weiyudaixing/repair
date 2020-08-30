"""This file contains all filters (kernels), i.e. icky-yucky image processing stuff."""

import numpy as np
import vigra
from torchy.utils import load_raw_data, get_filter_size


def get_filter_function(filter_name):
    if filter_name == 'Gaussian Smoothing':
        return vigra.gaussianSmoothing
    elif filter_name == 'Laplacian of Gaussian':
        return vigra.filters.laplacianOfGaussian
    elif filter_name == 'Hessian of Gaussian Eigenvalues':
        return vigra.filters.hessianOfGaussianEigenvalues
    elif filter_name == 'Gaussian Gradient Magnitude':
        return vigra.filters.gaussianGradientMagnitude
    else:
        raise NotImplementedError

def dummy_feature_prediciton(request):
    # load data slice in 
    print(request["roi_with_halo"])
    data = load_raw_data(request["data_filename"], request["roi_with_halo"])
    features = request["features"]
    sigmas = np.unique([f['sigma'] for f in features])
    filters = set([f['name'] for f in features])
    shape = list(data.shape)
    shape[0] = len(sigmas)
    shape[1] = np.sum([get_filter_size(f) for f in filters])
    output = np.empty(shape)
    for i, s in enumerate(sigmas):
        j = 0
        for f in filters:
            print("computing", f, s)
            if get_filter_size(f) == 1:
                output[i, j] = get_filter_function(f)(data[0, 0].astype(np.float32), s)
                j += 1
            else:
                feat = get_filter_function(f)(data[0, 0].astype(np.float32), s)
                for k in range(get_filter_size(f)):
                    output[i, j] = feat[...,k]
                    j += 1
    return output