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

from abc import ABC, abstractmethod
from typing import List, Any

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import TraceHessianRequest
from model_compression_toolkit.core.common.hessian.trace_hessian_config import TraceHessianConfig


class TraceHessianCalculator(ABC):
    """
       Abstract base class for a Hessian calculator.

       The Hessian matrix represents the second order partial derivatives
       of a function, and this class provides a structure to compute the
       Hessian for given inputs using a specified forward implementation.
    """

    def __init__(self,
                 graph: Graph,
                 trace_hessian_config: TraceHessianConfig,
                 input_images: List[Any],
                 fw_impl,
                 trace_hessian_request: TraceHessianRequest):
        """

        Args:
            graph (Graph): Graph representation of the float model for which Hessian is to be computed.
            trace_hessian_config (TraceHessianConfig): Configuration parameters for Hessian computation.
            input_images (List[Any]): List of input images for the hessian computation.
            fw_impl (FrameworkImplementation): Framework implementation used for the computation.
        """
        self.graph = graph
        self.hessian_config = trace_hessian_config
        self.input_images = input_images
        self.fw_impl = fw_impl
        self.hessian_request = trace_hessian_request

    @abstractmethod
    def compute(self):
        """
        Abstract method to compute the Hessian.
        Concrete implementations of this class should provide the method to
        compute the Hessian based on the initialized parameters.
        """
        raise NotImplemented
