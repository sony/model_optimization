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

import numpy as np
import torch
from torch import autograd
from tqdm import tqdm

from model_compression_toolkit.constants import HESSIAN_NUM_ITERATIONS, MIN_HESSIAN_ITER, HESSIAN_COMP_TOLERANCE
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import HessianScoresRequest, HessianScoresGranularity
from model_compression_toolkit.core.pytorch.back2framework.float_model_builder import FloatPyTorchModelBuilder
from model_compression_toolkit.core.pytorch.hessian.hessian_scores_calculator_pytorch import \
    HessianScoresCalculatorPytorch
from model_compression_toolkit.logger import Logger


class WeightsHessianScoresCalculatorPytorch(HessianScoresCalculatorPytorch):
    """
    Pytorch-specific implementation of the Hessian approximation scores computation w.r.t node's weights.
    """

    def __init__(self,
                 graph: Graph,
                 input_images: List[torch.Tensor],
                 fw_impl,
                 hessian_scores_request: HessianScoresRequest,
                 num_iterations_for_approximation: int = HESSIAN_NUM_ITERATIONS):
        """

        Args:
            graph: Computational graph for the float model.
            input_images: List of input images for the computation.
            fw_impl: Framework-specific implementation for Hessian scores computation.
            hessian_scores_request: Configuration request for which to compute the Hessian approximation scores.
            num_iterations_for_approximation: Number of iterations to use when approximating the Hessian scores.
        """

        super(WeightsHessianScoresCalculatorPytorch, self).__init__(graph=graph,
                                                                    input_images=input_images,
                                                                    fw_impl=fw_impl,
                                                                    hessian_scores_request=hessian_scores_request,
                                                                    num_iterations_for_approximation=num_iterations_for_approximation)

    def compute(self) -> List[np.ndarray]:
        """
        Compute the Hessian-based scores w.r.t target node's weights.
        The computed scores are returned in a numpy array. The shape of the result differs
        according to the requested granularity. If for example the node is Conv2D with a kernel
        shape of (2, 3, 3, 3) (namely, 3 input channels, 2 output channels and kernel size of 3x3)
        and the required granularity is HessianInfoGranularity.PER_TENSOR the result shape will be (1,),
        for HessianInfoGranularity.PER_OUTPUT_CHANNEL the shape will be (2,) and for
        HessianInfoGranularity.PER_ELEMENT a shape of (2, 3, 3, 3).

        Returns:
            The computed scores as a list of numpy ndarray for target node's weights.
            The function returns a list for compatibility reasons.
        """

        # Float model
        model, _ = FloatPyTorchModelBuilder(graph=self.graph).build_model()

        # Run model inference
        outputs = model(self.input_images)
        output_tensor = self.concat_tensors(outputs)
        device = output_tensor.device

        ipts_hessian_approx_scores = [torch.tensor([0.0],
                                                   requires_grad=True,
                                                   device=device)
                                     for _ in range(len(self.hessian_request.target_nodes))]

        prev_mean_results = None
        for j in tqdm(range(self.num_iterations_for_approximation)):
            # Getting a random vector with the same shape as the model output
            v = self._generate_random_vectors_batch(output_tensor.shape, device=device)
            f_v = torch.mean(torch.sum(v * output_tensor, dim=-1))
            for i, ipt_node in enumerate(self.hessian_request.target_nodes):  # Per Interest point weights tensor

                # Check if the target node's layer type is supported.
                if not ipt_node.is_kernel_op:
                    Logger.critical(f"Hessian information with respect to weights is not supported for "
                                    f"{ipt_node.type} layers.")  # pragma: no cover

                weights_tensor = getattr(getattr(model, ipt_node.name), ipt_node.kernel_attr)

                # Get the output channel index
                output_channel_axis = ipt_node.channel_axis.output
                shape_channel_axis = [i for i in range(len(weights_tensor.shape))]
                if self.hessian_request.granularity == HessianScoresGranularity.PER_OUTPUT_CHANNEL:
                    shape_channel_axis.remove(output_channel_axis)
                elif self.hessian_request.granularity == HessianScoresGranularity.PER_ELEMENT:
                    shape_channel_axis = ()

                # Compute gradients of f_v with respect to the weights
                f_v_grad = autograd.grad(outputs=f_v,
                                         inputs=weights_tensor,
                                         retain_graph=True)[0]

                # Trace{A^T * A} = sum of all squares values of A
                approx = f_v_grad ** 2
                if len(shape_channel_axis) > 0:
                    approx = torch.sum(approx, dim=shape_channel_axis)

                # Update node Hessian approximation mean over random iterations
                ipts_hessian_approx_scores[i] = (j * ipts_hessian_approx_scores[i] + approx) / (j + 1)

            # If the change to the maximal mean Hessian approximation is insignificant we stop the calculation
            # Note that we do not consider granularity when computing the mean
            if j > MIN_HESSIAN_ITER:
                if prev_mean_results is not None:
                    new_mean_res = torch.as_tensor([torch.mean(res) for res in ipts_hessian_approx_scores],
                                                   device=device)
                    relative_delta_per_node = (torch.abs(new_mean_res - prev_mean_results) /
                                               (torch.abs(new_mean_res) + 1e-6))
                    max_delta = torch.max(relative_delta_per_node)
                    if max_delta < HESSIAN_COMP_TOLERANCE:
                        break

            prev_mean_results = torch.as_tensor([torch.mean(res) for res in ipts_hessian_approx_scores], device=device)

        # Make sure all final shape are tensors and not scalar
        if self.hessian_request.granularity == HessianScoresGranularity.PER_TENSOR:
            ipts_hessian_approx_scores = [final_approx.reshape(1) for final_approx in ipts_hessian_approx_scores]

        # Add a batch axis to the Hessian approximation tensor (to align with the expected returned shape).
        # We assume per-image computation, so the batch axis size is 1.
        final_approx = [r_final_approx[np.newaxis, ...].detach().cpu().numpy()
                        for r_final_approx in ipts_hessian_approx_scores]

        return final_approx

