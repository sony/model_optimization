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

import torch
import numpy as np
from torch import nn

from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.gptq.common.gptq_graph import get_kernel_attribute_name_for_gptq
from mct_quantizers import PytorchQuantizationWrapper


class LinearTempDecay:
    """
    Annealing process for the soft quantizer regularization temperature term.
    """

    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 20, end_b: int = 2):
        """
        Initializes a LinearTempDecay object.

        Args:
            t_max: maximal time step.
            rel_start_decay: Decay step size at the beginning of the process.
            start_b: Starting value of the regularization term.
            end_b: Target value of the regularization term.
        """

        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t: float) -> float:
        """
        Cosine annealing scheduler for soft quantizer regularization temperature term.

        Args:
            t: The current time step.

        Returns: Scheduled temperature.
        """

        is_before_start_decay = (t < self.start_decay)

        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)

        return self.start_b * is_before_start_decay + \
               (1 - is_before_start_decay) * \
               (self.end_b + (self.start_b - self.end_b) * torch.maximum(to_torch_tensor(np.array([0.0])),
                                                                         to_torch_tensor(np.array((1 - rel_t)))))


class SoftQuantizerRegularization:
    """
    A class to handle the computation of soft quantizer regularization for GPTQ training.
    """

    def __init__(self, total_gradient_steps: int):
        """
        Initializes the regularization computation object with a LinearDecay object.

        Args:
            total_gradient_steps: The number of gradient steps during optimization.
        """

        # Initializing the temperature decay according to the number of expected gradient steps
        self.linear_decay = LinearTempDecay(total_gradient_steps)

        self.count_iter = 0

    def __call__(self, model: nn.Module, entropy_reg: float):
        """
        Returns the soft quantizer regularization value for SoftRounding.

        Args:
            model: A model to be quantized with SoftRounding.
            entropy_reg: Entropy value to scale the quantizer regularization.

        Returns: Regularization value.
        """

        soft_reg_aux: List[torch.Tensor] = []
        b = self.linear_decay(self.count_iter)
        for layer in model.modules():
            if isinstance(layer, PytorchQuantizationWrapper):
                kernel_attribute = get_kernel_attribute_name_for_gptq(layer_type=type(layer.layer),
                                                                      fw_info=DEFAULT_PYTORCH_INFO)

                st = layer.weights_quantizers[kernel_attribute].get_soft_targets()
                soft_reg_aux.append((1 - torch.pow(torch.abs(st - .5) * 2, b)).sum())

        reg = 0

        for sq in soft_reg_aux:
            reg += sq

        self.count_iter += 1

        return entropy_reg * reg
