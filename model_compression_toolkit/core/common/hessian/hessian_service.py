from typing import List, Any, Dict

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian.hessian_config import HessianConfig


class HessianService:
    def __init__(self,
                 # graph: Graph,
                 # hessian_configurations: List[HessianConfig],
                 # input_data: List[Any],
                 # hessian_compute_class: type
                 ):

        self.hessian_cfg_to_hessian_data = {}  # Dictionary to store Hessians by configuration
        self.hessian_configurations = []  # hessian_configurations
        self.input_data = None  # input_data
        self.graph = None  # graph
        self.fw_impl = None
        self.hessian_computes = []

    def set_graph(self, graph: Graph):
        self.graph = graph

    def set_fw_impl(self, fw_impl):
        self.fw_impl = fw_impl

    def add_hessian_configurations(self, hessian_configurations: List[HessianConfig]):
        self.hessian_configurations.extend(hessian_configurations)

    # def set_input_data(self, input_data: List[Any]):
    #     self.input_data = input_data

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

    def fetch_hessian(self, hessian_cfg:HessianConfig, input_images:List[Any]):
        if hessian_cfg in self.hessian_cfg_to_hessian_data:
            if id(input_images) in self.hessian_cfg_to_hessian_data[hessian_cfg]:
                return self.hessian_cfg_to_hessian_data[hessian_cfg][id(input_images)]

        self.compute(hessian_cfg, input_images)
        return self.hessian_cfg_to_hessian_data[hessian_cfg][id(input_images)]


hessian_service = HessianService()
