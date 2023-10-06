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

from typing import List


from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import TraceHessianRequest
from model_compression_toolkit.core.common.hessian.trace_hessian_config import TraceHessianConfig
from model_compression_toolkit.core.pytorch.hessian.trace_hessian_calculator_pytorch import \
    TraceHessianCalculatorPytorch
from model_compression_toolkit.logger import Logger
import torch

class WeightsTraceHessianCalculatorPytorch(TraceHessianCalculatorPytorch):
    """
    Pytorch-specific implementation of the Trace Hessian approximation computation w.r.t to a node's weights.
    """

    def __init__(self,
                 graph: Graph,
                 trace_hessian_config: TraceHessianConfig,
                 input_images: List[torch.Tensor],
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
        super(WeightsTraceHessianCalculatorPytorch, self).__init__(graph=graph,
                                                                 trace_hessian_config=trace_hessian_config,
                                                                 input_images=input_images,
                                                                 fw_impl=fw_impl,
                                                                 trace_hessian_request=trace_hessian_request)

    def compute(self):
        Logger.error(f"Hessian trace approx w.r.t weights is not supported for now")