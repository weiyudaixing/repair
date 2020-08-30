"""Torch filters for accelerated feature computation."""

import numpy as np

import h5py as h5

import torch
from torch.nn.functional import conv2d, conv3d
from torch.autograd.variable import Variable

# from dask.threaded import get
from dask.async import get_sync as get
# from dask.multiprocessing import get

from torchy.utils import timeit, reshape_volume_for_torch, no_lock
from threading import Lock

torch.set_num_threads(16)


def to_variable(tensor, device='cpu'):
    if isinstance(tensor, np.ndarray):
        if device == 'cpu':
            tensor = Variable(torch.from_numpy(tensor.astype('float32')))
        elif device == 'gpu':
            tensor = Variable(torch.from_numpy(tensor.astype('float32')).cuda())
    return tensor


class FeatureSuite(object):
    GAUSSIAN_KERNELS = {(2.5, 9): np.array([0.048297,
                                            0.08393,
                                            0.124548,
                                            0.157829,
                                            0.170793,
                                            0.157829,
                                            0.124548,
                                            0.08393,
                                            0.048297]),
                        (1.2, 9): np.array([0.001681,
                                            0.016844,
                                            0.087055,
                                            0.232853,
                                            0.323135,
                                            0.232853,
                                            0.087055,
                                            0.016844,
                                            0.001681]),
                        (0.6, 9): np.array([0,
                                            0.000015,
                                            0.006194,
                                            0.196119,
                                            0.595343,
                                            0.196119,
                                            0.006194,
                                            0.000015,
                                            0]),
                        (0.3, 9): np.array([0,
                                            0,
                                            0,
                                            0.04779,
                                            0.904419,
                                            0.04779,
                                            0,
                                            0,
                                            0]),
                        (0.3, 5): np.array([0,
                                            0.04779,
                                            0.904419,
                                            0.04779,
                                            0]),
                        (0.6, 5): np.array([0.006194,
                                            0.196125,
                                            0.595362,
                                            0.196125,
                                            0.006194]),
                        (5.0, 15): np.array([0.034619,
                                             0.044859,
                                             0.055857,
                                             0.066833,
                                             0.076841,
                                             0.084894,
                                             0.090126,
                                             0.09194,
                                             0.090126,
                                             0.084894,
                                             0.076841,
                                             0.066833,
                                             0.055857,
                                             0.044859,
                                             0.034619]),
                        (10.0, 15): np.array([0.057099,
                                              0.060931,
                                              0.064373,
                                              0.067333,
                                              0.06973,
                                              0.071493,
                                              0.072573,
                                              0.072936,
                                              0.072573,
                                              0.071493,
                                              0.06973,
                                              0.067333,
                                              0.064373,
                                              0.060931,
                                              0.057099]),
                        (1.0, 5): np.array([0.06136,
                                            0.24477,
                                            0.38774,
                                            0.24477,
                                            0.06136]),
                        (1.6, 9): np.array([0.011954,
                                            0.044953,
                                            0.115735,
                                            0.204083,
                                            0.246551,
                                            0.204083,
                                            0.115735,
                                            0.044953,
                                            0.011954]),
                        (3.5, 15): np.array([0.0161,
                                             0.027272,
                                             0.042598,
                                             0.061355,
                                             0.081488,
                                             0.099798,
                                             0.112705,
                                             0.117367,
                                             0.112705,
                                             0.099798,
                                             0.081488,
                                             0.061355,
                                             0.042598,
                                             0.027272,
                                             0.0161]),
                        (0.7, 5): np.array([0.01589,
                                            0.221542,
                                            0.525136,
                                            0.221542,
                                            0.01589])}

    DERIVATIVE_KERNEL = {None: np.array([-0.5, 0, 0.5])}

    SMALL_KERNEL_SIZE = 5
    MED_KERNEL_SIZE = 9
    BIG_KERNEL_SIZE = 15

    EPS = 0.0001
    ONE_BY_THREE = 0.3333333
    ONE_BY_SIX = 0.1666667
    ONE_BY_TWO = 0.5

    def __init__(self, ndim=2, num_workers=4, device='cpu', global_gpu_lock=None):
        assert ndim in [2, 3]
        self._global_gpu_lock = None
        self._eighess_on = None
        self._global_gpu_lock = global_gpu_lock
        # Assignments
        self.ndim = ndim
        self.num_workers = num_workers
        self.cache = {}
        self.device = device

    @property
    def global_gpu_lock(self):
        return no_lock() if self._global_gpu_lock is None or self.device == 'cpu' \
            else self._global_gpu_lock

    @global_gpu_lock.setter
    def global_gpu_lock(self, value):
        self._global_gpu_lock = value

    def activate_global_gpu_lock(self):
        self._global_gpu_lock = Lock()

    def deactivate_global_gpu_lock(self):
        self._global_gpu_lock = None

    def share_gpu_lock(self, other):
        other._global_gpu_lock = self._global_gpu_lock

    @property
    def conv(self):
        return conv2d if self.ndim == 2 else conv3d

    def stack_filters(self, *filters, **kwargs):
        convert_to_variable = kwargs.get('convert_to_variable', True)
        if self.ndim == 2:
            kernel_tensor = np.array([filter_.reshape(-1, 1)[None, ...] for filter_ in filters])
        else:
            kernel_tensor = np.array([filter_.reshape(-1, 1, 1)[None, ...] for filter_ in filters])
        kernel_tensor = self.to_variable(kernel_tensor) if convert_to_variable else kernel_tensor
        return kernel_tensor

    def pad_input(self, input_tensor, kernel_size):
        half_pad_size = kernel_size // 2
        pad_spec = [(0, 0), (0, 0), (half_pad_size, half_pad_size), (half_pad_size, half_pad_size)] + \
                   ([] if self.ndim == 2 else [(half_pad_size, half_pad_size)])
        padded_input_tensor = np.pad(input_tensor, pad_spec, 'reflect')
        return padded_input_tensor

    def sconv(self, input_tensor, kernel_tensor, padding=0, bias=None):
        # Separable convolutions
        # Get number of outputs
        num_outputs = kernel_tensor.size()[0]

        if self.ndim == 3:
            conved_012 = self.conv(input_tensor, kernel_tensor, padding=padding,
                                   bias=bias)
            conved_201 = self.conv(conved_012, kernel_tensor.permute(0, 1, 4, 2, 3),
                                   groups=num_outputs, bias=bias)
            conved_120 = self.conv(conved_201, kernel_tensor.permute(0, 1, 3, 4, 2),
                                   groups=num_outputs, bias=bias)
            output = conved_120
        else:
            conved_01 = self.conv(input_tensor, kernel_tensor, padding=padding)
            conved_10 = self.conv(conved_01, kernel_tensor.permute(0, 1, 3, 2), groups=num_outputs,
                                  bias=bias)
            output = conved_10
        return output

    def channel_to_batch(self, tensor):
        if self.ndim == 3:
            return tensor.permute(1, 0, 2, 3, 4)
        else:
            return tensor.permute(1, 0, 2, 3)

    def to_variable(self, tensor):
        return to_variable(tensor, device=self.device)

    def to_torch_tensor(self, numpy_tensor, device=None):
        device = self.device if device is None else device
        if device == 'gpu':
            return torch.from_numpy(numpy_tensor.astype('float32')).cuda()
        else:
            return torch.from_numpy(numpy_tensor.astype('float32'))

    def presmoothing(self, input_tensor):

        input_tensor_big = input_tensor_med = input_tensor_small = self.to_variable(input_tensor)

        small_kernel_sigmas = [0.3, 0.7, 1.0]
        med_kernel_sigmas = [1.6]
        big_kernel_sigmas = [3.5, 5.0, 10.0]

        small_kernel_padding = self.SMALL_KERNEL_SIZE // 2
        med_kernel_padding = self.MED_KERNEL_SIZE // 2
        big_kernel_padding = self.BIG_KERNEL_SIZE // 2

        # This is required because there's a bug in pytorch where group convolution requires
        # a bias which is not None. We define the bias once, transfer to GPU and cache it for
        # future use.
        if 'small_kernel_bias' not in self.cache.keys():
            small_kernel_bias = self.to_variable(np.zeros(shape=len(small_kernel_sigmas)))
            self.cache.update({'small_kernel_bias': small_kernel_bias})
        else:
            small_kernel_bias = self.cache['small_kernel_bias']

        # No bias for medium kernels (because there's just one feature, i.e. no grouped conv
        # required)
        if 'med_kernel_bias' not in self.cache.keys():
            med_kernel_bias = self.to_variable(np.zeros(shape=len(med_kernel_sigmas)))
            self.cache.update({'med_kernel_bias': med_kernel_bias})
        else:
            med_kernel_bias = self.cache['med_kernel_bias']

        med_kernel_bias = None if self.ndim == 2 else med_kernel_bias

        if 'big_kernel_bias' not in self.cache.keys():
            big_kernel_bias = self.to_variable(np.zeros(shape=len(big_kernel_sigmas)))
            self.cache.update({'big_kernel_bias': big_kernel_bias})
        else:
            big_kernel_bias = self.cache['big_kernel_bias']

        # No need to launch the kernel at every run
        if 'kernel_tensor_small' not in self.cache.keys():
            # Stack filters
            kernel_tensor_small = self.stack_filters(*[self.GAUSSIAN_KERNELS[(sig, self.SMALL_KERNEL_SIZE)]
                                                       for sig in small_kernel_sigmas])
            self.cache.update({'kernel_tensor_small': kernel_tensor_small})
        else:
            kernel_tensor_small = self.cache['kernel_tensor_small']

        if 'kernel_tensor_med' not in self.cache.keys():
            kernel_tensor_med = self.stack_filters(*[self.GAUSSIAN_KERNELS[(sig, self.MED_KERNEL_SIZE)]
                                                     for sig in med_kernel_sigmas])
            self.cache.update({'kernel_tensor_med': kernel_tensor_med})
        else:
            kernel_tensor_med = self.cache['kernel_tensor_med']

        if 'kernel_tensor_big' not in self.cache.keys():
            kernel_tensor_big = self.stack_filters(*[self.GAUSSIAN_KERNELS[(sig, self.BIG_KERNEL_SIZE)]
                                                     for sig in big_kernel_sigmas])
            self.cache.update({'kernel_tensor_big': kernel_tensor_big})
        else:
            kernel_tensor_big = self.cache['kernel_tensor_big']

        # Compute convolutions
        conved_small = self.sconv(input_tensor_small, kernel_tensor_small,
                                  padding=small_kernel_padding, bias=small_kernel_bias).data
        conved_med = self.sconv(input_tensor_med, kernel_tensor_med,
                                padding=med_kernel_padding, bias=med_kernel_bias).data
        conved_big = self.sconv(input_tensor_big, kernel_tensor_big,
                                padding=big_kernel_padding, bias=big_kernel_bias).data
        # Concatenate results and move to batch axis
        all_conved = self.channel_to_batch(torch.cat((conved_small, conved_med, conved_big), 1))
        return all_conved

    def d0(self, input_tensor):
        if 'one_channel_bias' in self.cache.keys():
            one_channel_bias = self.cache['one_channel_bias']
        else:
            one_channel_bias = self.to_variable(np.zeros(shape=(1,)))
            self.cache.update({'one_channel_bias': one_channel_bias})
        one_channel_bias = None if self.ndim == 2 else one_channel_bias
        # Gradient along axis 0
        input_tensor = Variable(input_tensor)
        kernel_tensor = self.stack_filters(self.DERIVATIVE_KERNEL[None])
        return self.sconv(input_tensor, kernel_tensor, padding=1, bias=one_channel_bias).data

    def d1(self, input_tensor):
        if 'one_channel_bias' in self.cache.keys():
            one_channel_bias = self.cache['one_channel_bias']
        else:
            one_channel_bias = self.to_variable(np.zeros(shape=(1,)))
            self.cache.update({'one_channel_bias': one_channel_bias})
        one_channel_bias = None if self.ndim == 2 else one_channel_bias
        # Gradient along axis 1
        kernel_tensor = self.stack_filters(self.DERIVATIVE_KERNEL[None])
        input_tensor = Variable(input_tensor)
        if self.ndim == 2:
            return self.sconv(input_tensor, kernel_tensor.permute(0, 1, 3, 2), padding=1,
                              bias=one_channel_bias).data
        else:
            return self.sconv(input_tensor, kernel_tensor.permute(0, 1, 3, 2, 4), padding=1,
                              bias=one_channel_bias).data

    def d2(self, input_tensor):
        if 'one_channel_bias' in self.cache.keys():
            one_channel_bias = self.cache['one_channel_bias']
        else:
            one_channel_bias = self.to_variable(np.zeros(shape=(1,)))
            self.cache.update({'one_channel_bias': one_channel_bias})
        # Gradient along axis 2
        kernel_tensor = self.stack_filters(self.DERIVATIVE_KERNEL[None])
        input_tensor = Variable(input_tensor)
        if self.ndim == 2:
            raise RuntimeError
        else:
            return self.sconv(input_tensor, kernel_tensor.permute(0, 1, 3, 4, 2), padding=1,
                              bias=one_channel_bias).data

    def dmag(self, *dns):
        if self.ndim == 3:
            d0, d1, d2 = dns[0], dns[1], dns[2]
            # No inplace ops, might cause threading bugs
            return torch.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        else:
            d0, d1 = dns[0], dns[1]
            return torch.sqrt(d0 * d0 + d1 * d1)

    def laplacian(self, *dnns):
        if self.ndim == 3:
            d00, d11, d22 = dnns[0], dnns[1], dnns[2]
            return d00 * d00 + d11 * d11 + d22 * d22
        else:
            d00, d11 = dnns[0], dnns[1]
            return d00 * d00 + d11 * d11

    def eighess(self, *dnms):
        if self.ndim == 3:
            d00, d01, d02, d11, d12, d22 = dnms[0], dnms[1], dnms[2], dnms[3], dnms[4], dnms[5]

            p1 = d01 * d01 + d02 * d02 + d12 * d12

            T = (d00 + d11 + d22) * self.ONE_BY_THREE

            d00_minus_T = d00 - T
            d11_minus_T = d11 - T
            d22_minus_T = d22 - T

            p2 = d00_minus_T * d00_minus_T + \
                 d11_minus_T * d11_minus_T + \
                 d22_minus_T * d22_minus_T + \
                 2. * p1

            p = torch.sqrt(p2 * self.ONE_BY_SIX)
            p_inv = (1./p)

            B00 = p_inv * d00_minus_T
            B01 = p_inv * d01
            B02 = p_inv * d02
            B11 = p_inv * d11_minus_T
            B12 = p_inv * d12
            B22 = p_inv * d22_minus_T

            detB = B00 * (B11 * B22 - B12 * B12) - \
                   B01 * (B01 * B22 - B12 * B02) + \
                   B02 * (B01 * B12 - B11 * B02)
            r = detB * self.ONE_BY_TWO

            phi = self.to_torch_tensor(np.zeros(shape=tuple(detB.size())), device=self._eighess_on)
            phi[r <= -1] = np.pi * self.ONE_BY_THREE
            phi[r >= 1] = 0.
            phi[-1 < r] = torch.acos(r[-1 < r]) * self.ONE_BY_THREE
            phi[r < 1] = torch.acos(r[r < 1]) * self.ONE_BY_THREE

            p_times_two = p * 2.
            eig1 = T + p_times_two * torch.cos(phi)
            eig3 = T + p_times_two * torch.cos(phi + ((2. * np.pi) * self.ONE_BY_THREE))
            eig2 = 3. * T - eig1 - eig3
            return torch.cat((eig1, eig2, eig3), 1)

        else:
            d00, d01, d11 = dnms[0], dnms[1], dnms[2]
            T = d00 + d11
            D = d00 * d11 - d01 * d01
            T_by_2 = T * self.ONE_BY_TWO
            K = torch.sqrt(T_by_2 * T_by_2 - D)
            eig1 = T_by_2 + K
            eig2 = T_by_2 - K
            return torch.cat((eig1, eig2), 1)

    def gpu_to_cpu(self, *inputs):
        return tuple(_input.cpu() for _input in inputs) if len(inputs) > 1 else inputs[0].cpu()

    def process_dsk_output(self, *input_features):
        if self.device == 'cpu':
            return torch.cat(input_features, 1).numpy()
        else:
            return torch.cat(input_features, 1).cpu().numpy()

    @property
    def dsk(self):
        if self.ndim == 2:
            # 2D
            _dsk = {'input': None,
                    'smooth': (self.presmoothing, 'input'),
                    'd0': (self.d0, 'smooth'),
                    'd1': (self.d1, 'smooth'),
                    'dmag': (self.dmag, 'd0', 'd1'),
                    'd00': (self.d0, 'd0'),
                    'd01': (self.d1, 'd0'),
                    'd11': (self.d1, 'd1'),
                    'laplacian': (self.laplacian, 'd00', 'd11'),
                    'eighess': (self.eighess, 'd00', 'd01', 'd11'),
                    'output': (self.process_dsk_output, 'smooth', 'dmag', 'laplacian', 'eighess')}
        else:
            # 3D
            _dsk = {'input': None,
                    'smooth': (self.presmoothing, 'input'),
                    'd0': (self.d0, 'smooth'),
                    'd1': (self.d1, 'smooth'),
                    'd2': (self.d2, 'smooth'),
                    'dmag': (self.dmag, 'd0', 'd1', 'd2'),
                    'd00': (self.d0, 'd0'),
                    'd01': (self.d1, 'd0'),
                    'd02': (self.d2, 'd0'),
                    'd11': (self.d1, 'd1'),
                    'd12': (self.d2, 'd1'),
                    'd22': (self.d2, 'd2'),
                    'laplacian': (self.laplacian, 'd00', 'd11', 'd22'),
                    'eighess': (self.eighess, 'd00', 'd01', 'd02', 'd11', 'd12', 'd22'),
                    'output': (self.process_dsk_output, 'smooth', 'dmag', 'laplacian', 'eighess')}
        return _dsk

    def compute_features(self, input_tensor):
        _dsk = self.dsk
        _dsk.update({'input': input_tensor})
        return get(_dsk, 'output', num_workers=self.num_workers)

    def remove_halo(self, tensor, halo_size):
        if self.ndim == 2:
            return tensor[:, :, halo_size:-halo_size, halo_size:-halo_size]
        else:
            return tensor[:, :, halo_size:-halo_size, halo_size:-halo_size, halo_size:-halo_size]

    def process_request(self, request):
        with h5.File(request['data_filename'], 'r+') as h5_file:
            input_tensor = reshape_volume_for_torch(h5_file["volume/data"][request['roi_with_halo']])
        # Compute features with a global lock
        with self.global_gpu_lock:
            feature_tensor = self.compute_features(input_tensor)
        feature_tensor = self.remove_halo(feature_tensor, request['halo_size'])
        return feature_tensor

    def _test_presmoothing(self, input_shape):
        input_array = np.random.uniform(size=input_shape)

        with timeit() as timestats:
            presmoothed = self.presmoothing(input_array)

        print("Input shape: {} || Output shape: {}".format(input_shape, presmoothed.size()))
        print("Elapsed time on {}: {}".format(self.device, timestats.elapsed_time))

    def _test_gradient(self, input_shape, wrt='0'):
        input_array = np.random.uniform(size=input_shape)
        # Presmooth
        presmoothed = self.presmoothing(input_array)
        # Get gradient func to test
        grad_func = getattr(self, "d{}".format(wrt))
        # Time gradient
        with timeit() as timestats:
            g = grad_func(presmoothed)

        print("Input shape: {} || Output shape: {}".format(presmoothed.size(), g.size()))
        print("Elapsed time: {}".format(timestats.elapsed_time))

    def _test_dmag_2d(self, input_shape):
        input_array = np.random.uniform(size=input_shape)
        # Presmooth
        presmoothed = self.presmoothing(input_array)
        # Compute gradients
        g0 = self.d0(presmoothed)
        g1 = self.d1(presmoothed)
        # Compute gradient magnitude
        with timeit() as timestats:
            gmag = self.dmag(g0, g1)

        print("Input shape: {} || Output shape: {}".format(presmoothed.size(), gmag.size()))
        print("Elapsed time: {}".format(timestats.elapsed_time))

    def _test_dmag_3d(self, input_shape):
        input_array = np.random.uniform(size=input_shape)
        # Presmooth
        presmoothed = self.presmoothing(input_array)
        # Compute gradients
        g0 = self.d0(presmoothed)
        g1 = self.d1(presmoothed)
        g2 = self.d1(presmoothed)
        # Compute gradient magnitude
        with timeit() as timestats:
            gmag = self.dmag(g0, g1, g2)

        print("Input shape: {} || Output shape: {}".format(presmoothed.size(), gmag.size()))
        print("Elapsed time: {}".format(timestats.elapsed_time))

    def _test_laplacian_2d(self, input_shape):
        input_array = np.random.uniform(size=input_shape)
        # Presmooth
        presmoothed = self.presmoothing(input_array)
        # Compute gradients
        g0 = self.d0(presmoothed)
        g1 = self.d1(presmoothed)
        g00 = self.d0(g0)
        g11 = self.d1(g1)
        # Compute and time laplacian
        with timeit() as timestats:
            lap = self.laplacian(g00, g11)

        print("Input shape: {} || Output shape: {}".format(presmoothed.size(), lap.size()))
        print("Elapsed time: {}".format(timestats.elapsed_time))

    def _test_eighess_2d(self, input_shape):
        input_array = np.random.uniform(size=input_shape)
        # Presmooth
        presmoothed = self.presmoothing(input_array)
        # Compute gradients
        g0 = self.d0(presmoothed)
        g1 = self.d1(presmoothed)
        g00 = self.d0(g0)
        g01 = self.d1(g0)
        g11 = self.d1(g1)

        # Compute and time laplacian
        with timeit() as timestats:
            eighess = self.eighess(g00, g01, g11)

        print("Input shape: {} || Output shape: {}".format(presmoothed.size(), eighess.size()))
        print("Elapsed time: {}".format(timestats.elapsed_time))

    def _test_eighess_3d(self, input_shape):
        input_array = np.random.uniform(size=input_shape)
        # Presmooth
        presmoothed = self.presmoothing(input_array)
        # Compute gradients
        g0 = self.d0(presmoothed)
        g1 = self.d1(presmoothed)
        g2 = self.d2(presmoothed)
        g00 = self.d0(g0)
        g01 = self.d1(g0)
        g02 = self.d2(g0)
        g11 = self.d1(g1)
        g12 = self.d2(g1)
        g22 = self.d2(g2)

        # Compute and time laplacian
        with timeit() as timestats:
            eighess = self.eighess(g00, g01, g02, g11, g12, g22)

        print("Input shape: {} || Output shape: {}".format(presmoothed.size(), eighess.size()))
        print("Elapsed time: {}".format(timestats.elapsed_time))

    def _test_dsk(self, input_shape):
        input_array = np.random.uniform(size=input_shape)
        print("[+] Starting dsk computation...")
        with timeit() as timestats:
            out = self.compute_features(input_array)
        print("[+] Dsk computation done...")
        print("Input shape: {} || Output shape: {}".format(input_shape, out.shape))
        print("Elapsed time: {}".format(timestats.elapsed_time))


if __name__ == '__main__':
    fs = FeatureSuite(ndim=3, num_workers=2, device='gpu')
    fs.activate_global_gpu_lock()

    # print("---- Testing Presmoothing ----")
    # fs._test_presmoothing((1, 1, 2000, 2000))
    #
    # print("---- Testing d0 ----")
    # fs._test_gradient((1, 1, 2000, 2000), wrt='0')
    #
    # print("---- Testing d1 ----")
    # fs._test_gradient((1, 1, 2000, 2000), wrt='1')

    # print("---- Testing dmag 2D ----")
    # fs._test_dmag_2d((1, 1, 2000, 2000))

    # print("---- Testing laplacian 2D ----")
    # fs._test_laplacian_2d((1, 1, 2000, 2000))

    # print("---- Testing eighess 2D ----")
    # fs._test_eighess_2d((1, 1, 2000, 2000))

    # print("---- Testing dsk 2D ----")
    # fs._test_dsk((1, 1, 2000, 2000))

    # print("---- Testing dsk 2D ----")
    # fs._test_dsk((1, 1, 2000, 2000))

    # print("---- Testing dsk 2D ----")
    # fs._test_dsk((1, 1, 2000, 2000))

    print("---- Testing Presmoothing 3D ----")
    fs._test_presmoothing((1, 1, 200, 200, 200))

    # print("---- Testing d0 3D ----")
    # fs._test_gradient((1, 1, 100, 100, 100), wrt='2')

    # print("---- Testing dmag 3D ----")
    # fs._test_dmag_3d((1, 1, 100, 100, 100))

    # print("---- Testing eighess3D ----")
    # fs._test_eighess_3d((1, 1, 100, 100, 100))

    print("---- Testing dsk 3D ----")
    fs._test_dsk((1, 1, 200, 200, 200))
