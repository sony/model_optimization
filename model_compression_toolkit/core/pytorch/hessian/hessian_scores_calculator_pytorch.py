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

from typing import Union, List

from model_compression_toolkit.constants import HESSIAN_NUM_ITERATIONS
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import HessianScoresRequest
from model_compression_toolkit.core.common.hessian.hessian_scores_calculator import HessianScoresCalculator
from model_compression_toolkit.logger import Logger
import torch


class HessianScoresCalculatorPytorch(HessianScoresCalculator):
    """
    Pytorch-specific implementation of the Hessian approximation scores Calculator.
    This class serves as a base for other Pytorch-specific Hessian approximation scores calculators.
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
            num_iterations_for_approximation: Number of iterations to use when approximating the Hessian based scores.

        """
        super(HessianScoresCalculatorPytorch, self).__init__(graph=graph,
                                                             input_images=input_images,
                                                             fw_impl=fw_impl,
                                                             hessian_scores_request=hessian_scores_request,
                                                             num_iterations_for_approximation=num_iterations_for_approximation)


    def concat_tensors(self, tensors_to_concate: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Concatenate model tensors into a single tensor.
        Args:
            tensors_to_concate: Tensors to concatenate.
        Returns:
            torch.Tensor of the concatenation of tensors.
        """
        _unfold_tensors = self.unfold_tensors_list(tensors_to_concate)
        _r_tensors = [torch.reshape(tensor, shape=[tensor.shape[0], -1]) for tensor in _unfold_tensors]

        concat_axis_dim = [o.shape[0] for o in _r_tensors]
        if not all(d == concat_axis_dim[0] for d in concat_axis_dim):
            Logger.critical(
                "Unable to concatenate tensors for gradient calculation due to mismatched shapes along the first axis.")  # pragma: no cover

        return torch.concat(_r_tensors, dim=1)
