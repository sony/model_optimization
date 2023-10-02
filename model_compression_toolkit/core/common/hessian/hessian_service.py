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

from typing import List, Any, Dict

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.hessian.hessian_config import HessianConfig


class HessianService:
    """
    A service to manage, store, and compute Hessian data.
    """
    def __init__(self):
        self.hessian_cfg_to_hessian_data = {}  # Dictionary to store Hessians by configuration and image list
        self._hessian_configurations = []  # hessian_configurations
        self.input_data = None  # input_data
        self.graph = None  # float graph
        self.fw_impl = None

    def set_graph(self, graph: Graph):
        """Set the float graph for the service."""
        self.graph = graph

    def set_fw_impl(self, fw_impl):
        """Set the framework implementation for the service."""
        self.fw_impl = fw_impl

    def add_hessian_configurations(self, hessian_configurations: List[HessianConfig]):
        """Extend the current list of Hessian configurations with additional configurations."""
        self._hessian_configurations.extend(hessian_configurations)

    def _set_hessian_configurations(self, hessian_configurations: List[HessianConfig]):
        """Set the list of Hessian configurations."""
        self._hessian_configurations = hessian_configurations

    def clear_cache(self):
        """Clear the cached Hessian data."""
        self.hessian_cfg_to_hessian_data={}

    def count_cache(self,
                    hessian_cfg: HessianConfig=None) -> int:
        """
        Count the cached Hessian data.
        If a specific configuration is provided, it counts for that configuration.
        Otherwise, it counts for all configurations.
        """
        if hessian_cfg:
            if hessian_cfg in self.hessian_cfg_to_hessian_data:
                return len(self.hessian_cfg_to_hessian_data[hessian_cfg])
            return 0
        return sum([len(x.values()) for x in self.hessian_cfg_to_hessian_data.values()])

    def compute(self,
                hessian_cfg:HessianConfig,
                input_images: List[Any]):
        """
        Compute the Hessian based on the provided configuration and input images.
        Store the computed Hessian in the cache.
        """
        if len(hessian_cfg.nodes_names_for_hessian_computation) == 1:
            # Only one compare point, nothing else to "weight"
            hessian = {hessian_cfg.nodes_names_for_hessian_computation[0]: 1.0}
        else:
            fw_hessian_calculator = self.fw_impl.get_framwork_hessian_calculator(hessian_cfg)
            hessian_calculator = fw_hessian_calculator(graph=self.graph,
                                                       config=hessian_cfg,
                                                       input_images=input_images,
                                                       fw_impl=self.fw_impl)
            hessian = hessian_calculator.compute()

        if hessian_cfg in self.hessian_cfg_to_hessian_data:
            self.hessian_cfg_to_hessian_data[hessian_cfg][id(input_images)] = hessian
        else:
            self.hessian_cfg_to_hessian_data[hessian_cfg] = {id(input_images): hessian}

    def fetch_hessian(self,
                      hessian_cfg:HessianConfig,
                      input_images:List[Any]=None) -> Dict[BaseNode, float]:
        """
        Fetch the Hessian for a given configuration and input images.
        If the Hessian isn't already computed, it will compute it on-the-fly.

        Args:
            hessian_cfg: Hessian configuration for the desired hessian data.
            input_images: Images to use as inputs for the float graph.

        Returns:
            Dictionary from interest point to hessian score.
        """
        if input_images is None:
            if hessian_cfg in self.hessian_cfg_to_hessian_data:
                return self.hessian_cfg_to_hessian_data[hessian_cfg]
            return {}

        if hessian_cfg in self.hessian_cfg_to_hessian_data:
            if id(input_images) in self.hessian_cfg_to_hessian_data[hessian_cfg]:
                return self.hessian_cfg_to_hessian_data[hessian_cfg][id(input_images)]

        # TODO: it may be better to compute different hessians before, or use other existing copmutations. So a phase of smarter computation and fetching can be added here

        # Computing the Hessian if it's not already available
        self.compute(hessian_cfg, input_images)
        return self.hessian_cfg_to_hessian_data[hessian_cfg][id(input_images)]


# Instantiating the Hessian service
hessian_service = HessianService()
