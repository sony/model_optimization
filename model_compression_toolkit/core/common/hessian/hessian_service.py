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

from typing import List, Any

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian.hessian_config import HessianConfig


class HessianService:
    def __init__(self):

        self.hessian_cfg_to_hessian_data = {}  # Dictionary to store Hessians by configuration and image list
        self._hessian_configurations = []  # hessian_configurations
        self.input_data = None  # input_data
        self.graph = None  # graph
        self.fw_impl = None
        self.hessian_computes = []

    def set_graph(self, graph: Graph):
        self.graph = graph

    def set_fw_impl(self, fw_impl):
        self.fw_impl = fw_impl

    def add_hessian_configurations(self, hessian_configurations: List[HessianConfig]):
        self._hessian_configurations.extend(hessian_configurations)

    def _set_hessian_configurations(self, hessian_configurations: List[HessianConfig]):
        self._hessian_configurations = hessian_configurations

    def clear_cache(self):
        self.hessian_cfg_to_hessian_data={}

    def _count_cache(self):
        return len([x.values() for x in self.hessian_cfg_to_hessian_data.values()])

    def compute(self, hessian_cfg:HessianConfig, input_images: List[Any]):
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

    def fetch_hessian(self, hessian_cfg:HessianConfig, input_images:List[Any]=None):
        if input_images is None:
            if hessian_cfg in self.hessian_cfg_to_hessian_data:
                return self.hessian_cfg_to_hessian_data[hessian_cfg]
            return {}

        if hessian_cfg in self.hessian_cfg_to_hessian_data:
            if id(input_images) in self.hessian_cfg_to_hessian_data[hessian_cfg]:
                return self.hessian_cfg_to_hessian_data[hessian_cfg][id(input_images)]

        self.compute(hessian_cfg, input_images)
        return self.hessian_cfg_to_hessian_data[hessian_cfg][id(input_images)]


hessian_service = HessianService()
