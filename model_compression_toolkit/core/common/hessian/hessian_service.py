# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
from typing import List, Any, Dict, Callable

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.hessian.hessian_config import HessianConfig
from model_compression_toolkit.core.common.hessian.hessian_request import HessianRequest
from model_compression_toolkit.logger import Logger


class HessianService:
    """
    A service to manage, store, and compute Hessian data.
    """

    def __init__(self,
                 graph: Graph,
                 representative_dataset: Callable,
                 hessian_configuration: HessianConfig,
                 fw_impl
                 ):

        self.graph = graph
        self.representative_dataset = representative_dataset
        # if not len(next(self.representative_dataset()))==len(self.graph.get_inputs()):
        #     Logger.error(f"Graph has {len(self.graph.get_inputs())} inputs, but representative dataset returns a list of {len(next(self.representative_dataset()))} inputs. Their length muse be identical.")
        self.hessian_configuration = hessian_configuration
        self.fw_impl = fw_impl

        self.hessian_request_to_score_list = {}  # Dictionary to store Hessians by configuration and image list

    def clear_cache(self):
        """Clear the cached Hessian data."""
        self.hessian_request_to_score_list={}

    def count_cache_of_request(self, hessian_request:HessianRequest) -> int:
        """
        Count the cached Hessian data.
        If a specific configuration is provided, it counts for that configuration.
        Otherwise, it counts for all configurations.
        """
        if hessian_request in self.hessian_request_to_score_list:
            return len(self.hessian_request_to_score_list[hessian_request])
        return 0

    def _sample_single_image_per_input(self):
        images = next(self.representative_dataset())
        assert isinstance(images, list)
        res = []
        for image in images:
            if image.shape[0]==1:
                res.append(image)
            else:
                res.append(np.expand_dims(image[0],0))
        for image in res:
            assert image.shape[0]==1
        return res

    def compute(self, hessian_request:HessianRequest):
        """
        Compute the Hessian based on the provided configuration and input images.
        Store the computed Hessian in the cache.
        """

        fw_hessian_calculator = self.fw_impl.get_framwork_hessian_calculator(hessian_request=hessian_request)
        images = self._sample_single_image_per_input()
        hessian_calculator = fw_hessian_calculator(graph=self.graph,
                                                   hessian_config=self.hessian_configuration,
                                                   input_images=images,
                                                   fw_impl=self.fw_impl,
                                                   hessian_request=hessian_request)
        hessian = hessian_calculator.compute()

        if hessian_request in self.hessian_request_to_score_list:
            self.hessian_request_to_score_list[hessian_request].append(hessian)
        else:
            self.hessian_request_to_score_list[hessian_request] = [hessian]

    def fetch_hessian(self, hessian_request: HessianRequest, required_size: int) -> Dict[BaseNode, float]:
        self._populate_cache_to_size(hessian_request, required_size)
        return self.hessian_request_to_score_list[hessian_request]


    def _populate_cache_to_size(self, hessian_request: HessianRequest, required_size: int):
        current_existing_hessians = self.count_cache_of_request(hessian_request)
        if required_size > current_existing_hessians:
            left_to_compute = required_size - current_existing_hessians
            for _ in range(left_to_compute):
                self.compute(hessian_request)


