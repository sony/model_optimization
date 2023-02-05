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
from torch import Tensor

from model_compression_toolkit.core.common.constants import RANGE_MAX, RANGE_MIN
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.qat.common.constants import FQ_MIN, FQ_MAX
from model_compression_toolkit.core.common import constants as C
from model_compression_toolkit import quantizers_infrastructure as qi, TrainingMethod
from model_compression_toolkit.qat.pytorch.quantizer.base_pytorch_qat_quantizer import BasePytorchQATTrainableQuantizer
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer
from model_compression_toolkit.quantizers_infrastructure.pytorch import inferable_quantizers as iq
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.qat.pytorch.quantizer.quantizer_utils import uniform_quantizer
from model_compression_toolkit.quantizers_infrastructure.common.trainable_quantizer_config import \
    TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig


@mark_quantizer(quantization_target=qi.QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.UNIFORM],
                quantizer_type=TrainingMethod.STE)
class STEUniformWeightQuantizer(BasePytorchQATTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize a layer inputs.
    """

    def __init__(self, quantization_config: TrainableQuantizerWeightsConfig):
        """
        Initialize a TrainableWeightQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: trainable quantizer config class
        """
        super().__init__(quantization_config)
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
                                layer: qi.PytorchQuantizationWrapper) -> Dict[str, nn.Parameter]:
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

        # Add min and max variables to layer.
        layer.register_parameter(name+"_"+FQ_MIN, nn.Parameter(to_torch_tensor(self.min_values), requires_grad=False))
        layer.register_parameter(name+"_"+FQ_MAX, nn.Parameter(to_torch_tensor(self.max_values), requires_grad=False))

        # Save the quantizer parameters for later calculations
        self.quantizer_parameters = {FQ_MIN: layer.get_parameter(name+"_"+FQ_MIN), FQ_MAX: layer.get_parameter(name+"_"+FQ_MAX)}

        return self.quantizer_parameters

    def __call__(self,
                 inputs: nn.Parameter,
                 training: bool) -> Tensor:
        """
        Quantize a tensor
        Args:
            inputs: Input tensor to quantize.
            training: whether in training mode or not
        Returns:
            quantized tensor
        """
        return uniform_quantizer(inputs, self.quantizer_parameters[FQ_MIN], self.quantizer_parameters[FQ_MAX], self.num_bits)

    def convert2inferable(self) -> iq.WeightsUniformInferableQuantizer:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            A pytorch inferable quanizer object.
        """
        _min = self.quantizer_parameters[FQ_MIN].cpu().detach().numpy()
        _max = self.quantizer_parameters[FQ_MAX].cpu().detach().numpy()

        return iq.WeightsUniformInferableQuantizer(num_bits=self.num_bits,
                                                   min_range=_min, max_range=_max,
                                                   per_channel=self.quantization_config.weights_per_channel_threshold,
                                                   channel_axis=self.quantization_config.weights_channels_axis)


@mark_quantizer(quantization_target=qi.QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.UNIFORM],
                quantizer_type=TrainingMethod.STE)
class STEUniformActivationQuantizer(BasePytorchQATTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize a layer activations.
    """

    def __init__(self, quantization_config: TrainableQuantizerActivationConfig):
        """
        Initialize a STEUniformActivationQuantizer object with parameters to use
        for uniform quantization.

        Args:
            quantization_config: trainable quantizer config class
        """
        super().__init__(quantization_config)

        np_min_range = quantization_config.activation_quantization_params[C.RANGE_MIN]
        np_max_range = quantization_config.activation_quantization_params[C.RANGE_MAX]
        self.min_range_tensor = torch.Tensor([np_min_range])
        self.max_range_tensor = torch.Tensor([np_max_range])
        self.num_bits = quantization_config.activation_n_bits
        self.quantizer_parameters = {}

    def initialize_quantization(self,
                                tensor_shape: torch.Size,
                                name: str,
                                layer: qi.PytorchQuantizationWrapper) -> Dict[str, nn.Parameter]:
        """
        Add min and max variables to layer.
        """
        layer.register_parameter(name+"_"+FQ_MIN, nn.Parameter(to_torch_tensor(self.min_range_tensor), requires_grad=True))
        layer.register_parameter(name+"_"+FQ_MAX, nn.Parameter(to_torch_tensor(self.max_range_tensor), requires_grad=True))

        # Save the quantizer parameters for later calculations
        self.quantizer_parameters = {FQ_MIN: layer.get_parameter(name+"_"+FQ_MIN), FQ_MAX: layer.get_parameter(name+"_"+FQ_MAX)}
        return self.quantizer_parameters

    def __call__(self,
                 inputs: torch.Tensor,
                 training: bool = True) -> torch.Tensor:
        """
        Quantize a tensor.
        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.

        Returns:
            The quantized tensor.
        """

        _min = self.quantizer_parameters[FQ_MIN]
        _max = self.quantizer_parameters[FQ_MAX]
        q_tensor = uniform_quantizer(inputs, _min, _max, self.num_bits)
        return q_tensor

    def convert2inferable(self) -> iq.ActivationUniformInferableQuantizer:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            A pytorch inferable quanizer object.
        """
        _min = self.quantizer_parameters[FQ_MIN].cpu().detach().numpy()
        _max = self.quantizer_parameters[FQ_MAX].cpu().detach().numpy()

        return iq.ActivationUniformInferableQuantizer(num_bits=self.num_bits,
                                                      min_range=_min, max_range=_max)
