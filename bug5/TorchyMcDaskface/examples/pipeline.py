from __future__ import print_function

from torchy.utils import timeit
from torchy.pipeline import dask_controller
from torchy.learning import learning


if __name__ == '__main__':

    P = dask_controller.Controller(num_workers=2)
    P.build_feature_computer_pool(num_computers=10, ndim=3, num_workers_per_computer=1,
                                  device='gpu')
    P.process_requests("datasets/sample_req*.json")

    with timeit() as time_info:
        output = P.get_results()
    print(output)
    print("Time Elapsed: {}".format(time_info.elapsed_time))

