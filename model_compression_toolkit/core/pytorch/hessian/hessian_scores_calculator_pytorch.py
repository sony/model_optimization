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

import torch

from model_compression_toolkit.core.common.hessian.hessian_scores_calculator import HessianScoresCalculator
from model_compression_toolkit.logger import Logger


class HessianScoresCalculatorPytorch(HessianScoresCalculator):
    """
    Pytorch-specific implementation of the Hessian approximation scores Calculator.
    This class serves as a base for other Pytorch-specific Hessian approximation scores calculators.
    """
    def _generate_random_vectors_batch(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Generate a batch of random vectors for Hutchinson estimation using Rademacher distribution.

        Args:
            shape: target shape.
            device: target device.

        Returns:
            Random tensor.
        """
        v = torch.randint(high=2, size=shape, device=device)
        v[v == 0] = -1
        return v

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
