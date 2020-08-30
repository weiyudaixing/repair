"""This file contains the dask pipeline creation and request handling API"""

from __future__ import print_function
from __future__ import division
import numpy as np
from dask.async import get_sync as get
# from dask.threaded import get

# import vigra
import json

import h5py
import os
from glob import glob
import itertools

from torchy.learning import learning
from torchy.filters import filters
from torchy.utils import pairwise
import torchy.filters.filtorch as filtorch

from threading import Lock
# from torchy.pipeline import writer


class Controller(object):
    """ 
    Controller class that does the dask book keeping and 
    transforms request to a dask graph
    """
    def __init__(self, options):
        self._feature_computer_pool = None
        self._feature_computer_pool_built = False
        self._max_edge_length = options.max_edge_length
        self._global_gpu_lock = Lock()
        self.num_workers = options.num_workers
        self.current_targets = []
        self.current_dsk = {}
        self.current_requests = None
        self.classifier_type = options.classifier_type
        self.output_file = None
        self.options = options

    def clear_dsk(self):
        self.current_targets = []
        self.current_dsk = {}
        self.current_requests = None

    def build_feature_computer_pool(self, num_computers=10, ndim=3, num_workers_per_computer=1,
                                    device='cpu'):
        """Build feature computer pool."""
        self._feature_computer_pool = [filtorch.FeatureSuite(num_workers=num_workers_per_computer,
                                                             ndim=ndim, device=device,
                                                             global_gpu_lock=self._global_gpu_lock)
                                       for _ in range(num_computers)]
        self._feature_computer_pool_built = True

    def extend_feature_computer_pool(self, by):
        """Add new computers to pool."""
        fcp = self.feature_computer_pool
        num_workers_per_computer = fcp[0].num_workers
        ndim = fcp[0].ndim
        device = fcp[0].device
        self._feature_computer_pool.extend([filtorch.FeatureSuite(num_workers=num_workers_per_computer,
                                                                  ndim=ndim, device=device,
                                                                  global_gpu_lock=self._global_gpu_lock)
                                            for _ in range(by)])

    @property
    def feature_computer_pool(self):
        return self._feature_computer_pool
 
    def process_requests(self, json_base_name):
        self.clear_dsk()
        requests = fetch_new_requests(json_base_name)
        dsk = {"RF": self.classifier_type}
        self.current_requests = requests

        if self.output_file is None:
            if_set = set([r["data_filename"] for r in requests])
            if len(if_set) > 1:
                raise NotImplementedError("Sorry: we can only handle one lane (single input file)")

            self.output_file = "results/full_prediction.h5"
            with h5py.File(self.output_file, "w") as f:
                with h5py.File(if_set.pop(), "r") as raw_f:
                    if self.options.rf_source == "from_request":
                        rf_file = requests[0]["classifier_filename"]
                    else:
                        rf_file = self.options.rf_source
                    n_classes = learning.count_labels(learning.get_classifier(self.classifier_type, filename=rf_file))
                    prediction_shape = [1, n_classes] + list(raw_f["volume/data"].shape)[:-1]
                f.create_dataset("data", prediction_shape)
            print("created output file %s" % self.output_file)

        if not self._feature_computer_pool_built:
            self.build_feature_computer_pool(num_computers=len(requests), ndim=3, device='cpu')
        
        requests = self.reduce_edge_size(requests)

        if len(requests) > len(self.feature_computer_pool):
            num_new_computers = len(requests) - len(self.feature_computer_pool)
            self.extend_feature_computer_pool(num_new_computers)

        for r, fc in zip(requests, self.feature_computer_pool):
            rid = r["req_id"]
            r["output_file_name"] = self.output_file
            dsk["request-%i" % rid] = r
            dsk["RF-File-%i" % rid] = r["classifier_filename"]
            dsk["computed-features-%i" % rid] = (fc.process_request, "request-%i" % rid)
            dsk["predicted-image-%i"%rid] = (learning.image_prediction, "RF", "RF-File-%i" % rid, 'computed-features-%i' % rid, "request-%i" % rid)
            # dsk["write-%i"%rid] = (writer.write_output, "predicted-image-%i"%rid, "request-%i"%rid)
            self.current_targets.append("predicted-image-%i"%rid)
        self.current_dsk = dsk

    def reduce_edge_size(self, requests):
        if len(requests) == 0:
            return
        ndim = requests[0]["feature_dim"]
        for i,r in enumerate(requests):
            edge_lengths = [((r["cutout"][0]["%smax"%c]-r["cutout"][0]["%smin"%c]) // self._max_edge_length,
                              r["cutout"][0]["%smin"%c], r["cutout"][0]["%smax"%c]) for c in "xyz"[:ndim]]

            splits = []
            l = edge_lengths[0]
            # generate the min/ max grid bounds
            for m in itertools.product(*[pairwise(np.linspace(l[1], l[2], num=l[0]+2, dtype=int)) for l in edge_lengths]):
                r_slice = r.copy()
                for k, c in enumerate("xyz"[:ndim]):
                    r["cutout"][0]["%smin"%c] = m[k][0]
                    r["cutout"][0]["%smax"%c] = m[k][1]
                cutout_to_slice(r_slice)
                splits.append(r_slice)

            # dark python magic to flatten the nested list
            requests[i] = splits
            flatten = lambda *args: list(result for mid in args for result in (flatten(*mid)
                                                if isinstance(mid, (tuple, list)) else (mid,)))
            return flatten(requests)

    def get_results(self):
        return get(self.current_dsk, self.current_targets, num_workers=self.num_workers)

    def get_request(self):
        return self.current_requests


def fetch_new_requests(base_name):
    requests = []
    for json_file_name in glob(base_name):
        with open(json_file_name) as data_file:    
            data = json.load(data_file)
            data["halo_size"] = 20
            cutout_to_slice(data)
            requests.append(data)
    return requests


def m(xyz):
    # limit the halo size if halo would surpass the boundary
    return max(0, xyz)


def cutout_to_slice(request):
    """
    conversion function from request format ROI to list of python slices
    that follow the data format convention (batch, channel, x, y, z)
    halo_size: number of pixels that are added to every boundary
    """

    c = request["cutout"][0]
    hs = request["halo_size"]

    if len(c)/2 == 2:
        request["roi_with_halo"] = (slice(None), slice(None), 
                slice(m(c['xmin']-hs),c['xmax']+hs),
                slice(m(c['ymin']-hs),c['ymax']+hs), 0)
        request["roi_output"] = (slice(None), slice(None), 
                slice(m(c['xmin']),c['xmax']),
                slice(m(c['ymin']),c['ymax']))

    if len(c)/2 == 3:
        request["roi_with_halo"] = (slice(m(c['xmin']-hs),c['xmax']+hs),
                slice(m(c['ymin']-hs),c['ymax']+hs),
                slice(m(c['zmin']-hs),c['zmax']+hs), 0)
        request["roi_output"] = (slice(None), slice(None), 
                slice(m(c['xmin']),c['xmax']),
                slice(m(c['ymin']),c['ymax']),
                slice(m(c['zmin']),c['zmax']))


