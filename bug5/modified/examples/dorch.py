import numpy as np
from dask.multiprocessing import get

from torchy.utils import timeit

import torch
from torch.nn.functional import conv2d
from torch.autograd import Variable


def conv(input, kernel):
    print("[>] [conv]")
    if isinstance(input, np.ndarray):
        input = Variable(torch.from_numpy(input))
    if isinstance(kernel, np.ndarray):
        kernel = Variable(torch.from_numpy(kernel))
    out = conv2d(input=input, weight=kernel).data
    print("[<] [conv]")
    return out


def make_zeros(fid, shape):
    print("[>] [make_zeros::fid_{}]".format(fid))
    print("[<] [make_zeros::fid_{}]".format(fid))
    return np.zeros(shape=shape)


if __name__ == '__main__':
    dsk = {'input': (np.zeros, (1, 1, 100, 100)),
           'weights': (np.zeros, (10, 1, 5, 5)),
           'conv1': (conv, 'input', 'weights')}

    with timeit() as time_info:
        output = get(dsk, 'conv1')

    print("Time Elapsed: {}".format(time_info.elapsed_time))

