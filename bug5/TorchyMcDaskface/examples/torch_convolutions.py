import numpy as np

import torch
from torch.nn.functional import conv2d, conv3d
from torch.autograd.variable import Variable

from torchy.utils import timeit


def conv(input_size, kernel_size, num_output=1, num_input=1):
    input_variable = Variable(torch.from_numpy(np.random.uniform(size=(num_input, 1,
                                                                       input_size[0],
                                                                       input_size[1]))))
    kernel_variable = Variable(torch.from_numpy(np.random.uniform(size=(num_output,
                                                                        1,
                                                                        kernel_size[0],
                                                                        kernel_size[1]))))

    # bias = Variable(torch.from_numpy(np.zeros(shape=(num_output,))))

    with timeit() as timestat:
        # output = conv2d(input_variable, kernel_variable, bias=bias, groups=num_output)
        output = conv2d(input_variable, kernel_variable)

    print("Input size: {} || Kernel size: {} || Num outputs: {}".format(input_size, kernel_size,
                                                                        num_output))
    print("Output shape: {}".format(output.data.size()))
    print("Elapsed time: {}".format(timestat.elapsed_time))


if __name__ == '__main__':
    # print(tit.timeit('conv((100, 100), (9, 1))', globals=globals(), number=100))
    conv((100, 100), (9, 1), 10, 10)