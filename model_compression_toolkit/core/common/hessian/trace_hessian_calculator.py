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
    Abstract base class for computing an approximation of the trace of the Hessian.

    This class provides a structure for implementing different methods to compute
    the trace of the Hessian approximation based on the provided configuration,
    input images, and other parameters.
    """

    def __init__(self,
                 graph: Graph,
                 trace_hessian_config: TraceHessianConfig,
                 input_images: List[Any],
                 fw_impl,
                 trace_hessian_request: TraceHessianRequest):
        """
        Args:
            graph: Computational graph for the float model.
            trace_hessian_config: Configuration for the approximation of the trace of the Hessian.
            input_images: List of input images for the computation.
            fw_impl: Framework-specific implementation for trace Hessian computation.
            trace_hessian_request: Configuration request for which to compute the trace Hessian approximation.
        """
        self.graph = graph
        self.hessian_config = trace_hessian_config
        self.input_images = input_images
        self.fw_impl = fw_impl
        self.hessian_request = trace_hessian_request

    @abstractmethod
    def compute(self) -> List[float]:
        """
        Abstract method to compute the approximation of the trace of the Hessian.

        This method should be implemented by subclasses to provide the specific
        computation method for the trace Hessian approximation.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement compute method.')  # pragma: no cover
