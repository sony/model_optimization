from abc import ABC, abstractmethod
from typing import List, Any, Dict

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian.hessian_config import HessianConfig


class HessianCalculator(ABC):

    def __init__(self,
                 graph: Graph,
                 config: HessianConfig,
                 input_images: List[Any],
                 fw_impl):
        self.graph = graph
        self.config = config
        self.input_images = input_images
        self.fw_impl = fw_impl

    @abstractmethod
    def compute(self):
        raise NotImplemented
