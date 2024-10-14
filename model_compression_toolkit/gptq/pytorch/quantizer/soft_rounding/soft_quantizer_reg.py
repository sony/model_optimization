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
from typing import List, Callable

import torch
from torch import nn

from mct_quantizers import PytorchQuantizationWrapper
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.gptq.common.gptq_graph import get_kernel_attribute_name_for_gptq


class SoftQuantizerRegularization:
    """
    A class to handle the computation of soft quantizer regularization for GPTQ training.
    """

    def __init__(self, beta_scheduler: Callable[[int], float]):
        """
        Initializes the regularization computation object with a LinearDecay object.

        Args:
            beta_scheduler: a callable that accepts current time step and returns a corresponding beta value.
        """

        # Initializing the temperature decay according to the number of expected gradient steps
        self.beta_scheduler = beta_scheduler

        self.count_iter = 0

    def __call__(self, model: nn.Module, entropy_reg: float, layer_weights: torch.Tensor):
        """
        Returns the soft quantizer regularization value for SoftRounding.

        Args:
            model: A model to be quantized with SoftRounding.
            entropy_reg: Entropy value to scale the quantizer regularization.
            layer_weights: a vector of layers weights.

        Returns: Regularization value.
        """
        layers = [m for m in model.modules() if isinstance(m, PytorchQuantizationWrapper)]

        if layer_weights.shape[0] != len(layers):
            raise ValueError(f'Expected weights.shape[0] to be {len(layers)}, '
                             f'received shape {layer_weights.shape}.')    # pragma: no cover
        max_w = layer_weights.max()

        b = self.beta_scheduler(self.count_iter)
        reg = 0
        for layer, w in zip(layers, layer_weights):
            kernel_attribute = get_kernel_attribute_name_for_gptq(layer_type=type(layer.layer),
                                                                  fw_info=DEFAULT_PYTORCH_INFO)

            st = layer.weights_quantizers[kernel_attribute].get_soft_targets()
            soft_loss = (1 - torch.pow(torch.abs(st - .5) * 2, b)).sum()
            reg += w * soft_loss

        reg = reg / max_w
        self.count_iter += 1

        return entropy_reg * reg
