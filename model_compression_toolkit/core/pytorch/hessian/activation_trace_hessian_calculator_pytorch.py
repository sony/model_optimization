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

from torch import autograd
from tqdm import tqdm

from model_compression_toolkit.constants import MIN_HESSIAN_ITER, HESSIAN_COMP_TOLERANCE, HESSIAN_NUM_ITERATIONS
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import TraceHessianRequest, HessianInfoGranularity
from model_compression_toolkit.core.pytorch.back2framework.float_model_builder import FloatPyTorchModelBuilder
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

            model_output_nodes = [ot.node for ot in self.graph.get_outputs()]

            if self.hessian_request.target_node in model_output_nodes:
                Logger.critical("Activation Hessian approximation cannot be computed for model outputs. Exclude output nodes from Hessian request targets.")
            grad_model_outputs = [self.hessian_request.target_node] + model_output_nodes
            model, _ = FloatPyTorchModelBuilder(graph=self.graph, append2output=grad_model_outputs).build_model()
            model.eval()

            # Run model inference
            # Set inputs to track gradients during inference
            for input_tensor in self.input_images:
                input_tensor.requires_grad_()
                input_tensor.retain_grad()

            outputs = model(*self.input_images)

            if len(outputs) != len(grad_model_outputs):
                Logger.critical(f"Mismatch in expected and actual model outputs for activation Hessian approximation. Expected {len(grad_model_outputs)} outputs, received {len(outputs)}.")

            # Extracting the intermediate activation tensors and the model real output
            # TODO: we are assuming that the hessian request is for a single node.
            #  When we extend it to multiple nodes in the same request, then we should modify this part to take
            #  the first "num_target_nodes" outputs from the output list.
            #  We also assume that the target nodes are not part of the model output nodes, if this assumption changed,
            #  then the code should be modified accordingly.
            target_activation_tensors = [outputs[0]]
            output_tensors = outputs[1:]
            device = output_tensors[0].device

            # Concat outputs
            # First, we need to unfold all outputs that are given as list, to extract the actual output tensors
            output = self.concat_tensors(output_tensors)

            ipts_hessian_trace_approx = []
            for ipt_tensor in tqdm(target_activation_tensors):  # Per Interest point activation tensor
                trace_hv = []
                for j in range(self.num_iterations_for_approximation):  # Approximation iterations
                    # Getting a random vector with normal distribution
                    v = torch.randn(output.shape, device=device)
                    f_v = torch.sum(v * output)

                    # Computing the hessian trace approximation by getting the gradient of (output * v)
                    hess_v = autograd.grad(outputs=f_v,
                                          inputs=ipt_tensor,
                                          retain_graph=True,
                                          allow_unused=True)[0]
                    if hess_v is None:
                        # In case we have an output node, which is an interest point, but it is not differentiable,
                        # we still want to set some weight for it. For this, we need to add this dummy tensor to the ipt
                        # Hessian traces list.
                        trace_hv.append(torch.tensor([0.0],
                                                     requires_grad=True,
                                                     device=device))
                        break
                    hessian_trace_approx = torch.sum(torch.pow(hess_v, 2.0))

                    # If the change to the mean Hessian approximation is insignificant we stop the calculation
                    if j > MIN_HESSIAN_ITER:
                        new_mean = torch.mean(torch.stack([hessian_trace_approx, *trace_hv]))
                        delta = new_mean - torch.mean(torch.stack(trace_hv))
                        if torch.abs(delta) / (torch.abs(new_mean) + 1e-6) < HESSIAN_COMP_TOLERANCE:
                            trace_hv.append(hessian_trace_approx)
                            break

                    trace_hv.append(hessian_trace_approx)

                ipts_hessian_trace_approx.append(torch.mean(torch.stack(trace_hv)))  # Get averaged Hessian trace approximation

            # If a node has multiple outputs, it means that multiple approximations were computed
            # (one per output since granularity is per-tensor). In this case we average the approximations.
            if len(ipts_hessian_trace_approx) > 1:
                # Stack tensors and compute the average
                ipts_hessian_trace_approx = [torch.stack(ipts_hessian_trace_approx).mean()]

            ipts_hessian_trace_approx = torch_tensor_to_numpy(torch.Tensor(
                ipts_hessian_trace_approx))  # Just to get one tensor instead of list of tensors with single element

            return ipts_hessian_trace_approx.tolist()

        else:
            Logger.critical(f"PyTorch activation Hessian's trace approximation does not support {self.hessian_request.granularity} granularity.")

