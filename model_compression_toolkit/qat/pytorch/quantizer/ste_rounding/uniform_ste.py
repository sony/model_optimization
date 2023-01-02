# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from model_compression_toolkit.core.common.constants import RANGE_MAX, RANGE_MIN
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig
from model_compression_toolkit.qat.common.constants import FQ_MIN, FQ_MAX
from model_compression_toolkit import qunatizers_infrastructure as qi
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.qunatizers_infrastructure.pytorch.quantizer_utils import uniform_quantizer


class STEUniformWeightQuantizer(qi.BasePytorchQuantizer):
    """
    Trainable constrained quantizer to quantize a layer inputs.
    """

    def __init__(self, quantization_config: NodeWeightsQuantizationConfig):
        """
        Initialize a TrainableWeightQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: node quantization config class
        """
        super().__init__(quantization_config,
                         qi.QuantizationTarget.Weights,
                         [qi.QuantizationMethod.UNIFORM])
        self.num_bits = self.quantization_config.weights_n_bits
        self.min_int = 0
        self.max_int = 2 ** self.num_bits - 1
        self.max_values = quantization_config.weights_quantization_params[RANGE_MAX]
        self.min_values = quantization_config.weights_quantization_params[RANGE_MIN]
        self.min_max_shape = np.asarray(self.max_values).shape
        self.max = np.reshape(self.max_values,
                              [-1]) if self.quantization_config.weights_per_channel_threshold else float(
            self.max_values)
        self.min = np.reshape(self.min_values,
                              [-1]) if self.quantization_config.weights_per_channel_threshold else float(
            self.min_values)

        self.quantizer_parameters = {}

    def initialize_quantization(self,
                                tensor_shape: torch.Size,
                                name: str,
                                layer: nn.Module) -> Dict[str, nn.Parameter]:
        """
        Add min and max variables to layer.
        Args:
            tensor_shape: Tensor shape the quantizer quantize.
            name: Prefix of variables names.
            layer: Layer to add the variables to. The variables are saved
            in the layer's scope.

        Returns:
            Dictionary of new variables.
        """
        _shape = len(self.min) if self.quantization_config.weights_channels_axis else ()
        _val = torch.ones(_shape) * self.min
        self.fq_min = nn.Parameter(to_torch_tensor(_val), requires_grad=False)

        _shape = len(self.max) if self.quantization_config.weights_channels_axis else ()
        _val = torch.ones(_shape) * self.max
        self.fq_max = nn.Parameter(to_torch_tensor(_val), requires_grad=False)

        # save the quantizer added parameters for later calculations
        self.quantizer_parameters = {FQ_MIN: self.fq_min, FQ_MAX: self.fq_max}

        return self.quantizer_parameters

    def __call__(self,
                 inputs: nn.Parameter,
                 training: bool) -> nn.Parameter:
        """
        Quantize a tensor
        Args:
            inputs: Input tensor to quantize.
            training: whether in training mode or not
        Returns:
            quantized tensor
        """
        return uniform_quantizer(inputs, self.min_values, self.max_values, self.num_bit)
