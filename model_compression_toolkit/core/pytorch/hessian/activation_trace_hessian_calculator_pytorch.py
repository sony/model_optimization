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

from typing import List, Tuple, Dict, Any

from torch import autograd
from tqdm import tqdm
import numpy as np

from model_compression_toolkit.constants import MIN_JACOBIANS_ITER, JACOBIANS_COMP_TOLERANCE, HESSIAN_NUM_ITERATIONS
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import TraceHessianRequest, HessianInfoGranularity
from model_compression_toolkit.core.pytorch.hessian.pytorch_model_gradients import PytorchModelGradients
from model_compression_toolkit.core.pytorch.hessian.trace_hessian_calculator_pytorch import \
    TraceHessianCalculatorPytorch
from model_compression_toolkit.core.pytorch.utils import torch_tensor_to_numpy
from model_compression_toolkit.logger import Logger
import torch

class ActivationTraceHessianCalculatorPytorch(TraceHessianCalculatorPytorch):
    """
    Pytorch implementation of the Trace Hessian approximation Calculator for activations.
    """
    def __init__(self,
                 graph: Graph,
                 input_images: List[torch.Tensor],
                 fw_impl,
                 trace_hessian_request: TraceHessianRequest,
                 num_iterations_for_approximation: int = HESSIAN_NUM_ITERATIONS):
        """
        Args:
            graph: Computational graph for the float model.
            input_images: List of input images for the computation.
            fw_impl: Framework-specific implementation for trace Hessian approximation computation.
            trace_hessian_request: Configuration request for which to compute the trace Hessian approximation.
            num_iterations_for_approximation: Number of iterations to use when approximating the Hessian trace.

        """
        super(ActivationTraceHessianCalculatorPytorch, self).__init__(graph=graph,
                                                                      input_images=input_images,
                                                                      fw_impl=fw_impl,
                                                                      trace_hessian_request=trace_hessian_request,
                                                                      num_iterations_for_approximation=num_iterations_for_approximation)

    def compute(self) -> List[float]:
        """
        Compute the approximation of the trace of the Hessian w.r.t a node's activations.

        Returns:
            List[float]: Approximated trace of the Hessian for an interest point.
        """
        if self.hessian_request.granularity == HessianInfoGranularity.PER_TENSOR:
            # Set inputs to require_grad
            for input_tensor in self.input_images:
                input_tensor.requires_grad_()

            model_grads_net = PytorchModelGradients(graph_float=self.graph,
                                                    trace_hessian_request=self.hessian_request
                                                    )

            # Run model inference
            output_tensors = model_grads_net(self.input_images)
            device = output_tensors[0].device

            # Concat outputs
            # First, we need to unfold all outputs that are given as list, to extract the actual output tensors
            output = self._concat_tensors(output_tensors)

            ipts_jac_trace_approx = []
            for ipt in tqdm(model_grads_net.interest_points_tensors):  # Per Interest point activation tensor
                trace_jv = []
                for j in range(self.num_iterations_for_approximation):  # Approximation iterations
                    # Getting a random vector with normal distribution
                    v = torch.randn(output.shape, device=device)
                    f_v = torch.sum(v * output)

                    # Computing the hessian trace approximation by getting the gradient of (output * v)
                    jac_v = autograd.grad(outputs=f_v,
                                          inputs=ipt,
                                          retain_graph=True,
                                          allow_unused=True)[0]
                    if jac_v is None:
                        # In case we have an output node, which is an interest point, but it is not differentiable,
                        # we still want to set some weight for it. For this, we need to add this dummy tensor to the ipt
                        # jacobian traces list.
                        trace_jv.append(torch.tensor([0.0],
                                                     requires_grad=True,
                                                     device=device))
                        break
                    jac_v = torch.reshape(jac_v, [jac_v.shape[0], -1])
                    jac_trace_approx = torch.mean(torch.sum(torch.pow(jac_v, 2.0)))

                    # If the change to the mean Jacobian approximation is insignificant we stop the calculation
                    if j > MIN_JACOBIANS_ITER:
                        new_mean = torch.mean(torch.stack([jac_trace_approx, *trace_jv]))
                        delta = new_mean - torch.mean(torch.stack(trace_jv))
                        if torch.abs(delta) / (torch.abs(new_mean) + 1e-6) < JACOBIANS_COMP_TOLERANCE:
                            trace_jv.append(jac_trace_approx)
                            break

                    trace_jv.append(jac_trace_approx)
                ipts_jac_trace_approx.append(2 * torch.mean(torch.stack(trace_jv)) / output.shape[
                    -1])  # Get averaged jacobian trace approximation

            # If a node has multiple outputs, it means that multiple approximations were computed
            # (one per output since granularity is per-tensor). In this case we average the approximations.
            if len(ipts_jac_trace_approx)>1:
                # Stack tensors and compute the average
                ipts_jac_trace_approx = [torch.stack(ipts_jac_trace_approx).mean()]

            ipts_jac_trace_approx = torch_tensor_to_numpy(torch.Tensor(
                ipts_jac_trace_approx))  # Just to get one tensor instead of list of tensors with single element

            return ipts_jac_trace_approx.tolist()

        else:
            Logger.error(f"{self.hessian_request.granularity} is not supported for Pytorch activation hessian's trace approx calculator")

