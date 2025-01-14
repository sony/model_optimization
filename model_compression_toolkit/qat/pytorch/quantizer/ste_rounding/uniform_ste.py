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
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from model_compression_toolkit.constants import RANGE_MAX, RANGE_MIN
from model_compression_toolkit.trainable_infrastructure.common.constants import FQ_MIN, FQ_MAX

from mct_quantizers import QuantizationMethod
from mct_quantizers import QuantizationTarget, PytorchQuantizationWrapper

from model_compression_toolkit.qat.pytorch.quantizer.base_pytorch_qat_weight_quantizer import BasePytorchQATWeightTrainableQuantizer
from mct_quantizers import mark_quantizer
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.trainable_infrastructure import TrainingMethod
from model_compression_toolkit.trainable_infrastructure.pytorch.quantizer_utils import uniform_quantizer
from mct_quantizers.pytorch.quantizers import \
    WeightsUniformInferableQuantizer
from model_compression_toolkit.trainable_infrastructure.common.trainable_quantizer_config import \
    TrainableQuantizerWeightsConfig
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.UNIFORM],
                identifier=TrainingMethod.STE)
class STEUniformWeightQATQuantizer(BasePytorchQATWeightTrainableQuantizer):
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

    def initialize_quantization(self,
                                tensor_shape: torch.Size,
                                name: str,
                                layer: PytorchQuantizationWrapper):
        """
        Add quantizer parameters to the quantizer parameters dictionary

        Args:
            tensor_shape: tensor shape of the quantized tensor.
            name: Tensor name.
            layer: Layer to quantize.
        """

        # Add min and max variables to layer.
        layer.register_parameter(name+"_"+FQ_MIN, nn.Parameter(to_torch_tensor(self.min_values), requires_grad=False))
        layer.register_parameter(name+"_"+FQ_MAX, nn.Parameter(to_torch_tensor(self.max_values), requires_grad=False))

        # Save the quantizer parameters for later calculations
        self.add_quantizer_variable(FQ_MIN, layer.get_parameter(name+"_"+FQ_MIN), VariableGroup.QPARAMS)
        self.add_quantizer_variable(FQ_MAX, layer.get_parameter(name+"_"+FQ_MAX), VariableGroup.QPARAMS)

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
        return uniform_quantizer(inputs, self.get_quantizer_variable(FQ_MIN), self.get_quantizer_variable(FQ_MAX), self.num_bits)

    def convert2inferable(self) -> WeightsUniformInferableQuantizer:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            A pytorch inferable quanizer object.
        """
        _min = self.get_quantizer_variable(FQ_MIN).cpu().detach().numpy()
        _max = self.get_quantizer_variable(FQ_MAX).cpu().detach().numpy()

        return WeightsUniformInferableQuantizer(num_bits=self.num_bits,
                                                min_range=_min.tolist(),
                                                max_range=_max.tolist(),
                                                per_channel=self.quantization_config.weights_per_channel_threshold,
                                                channel_axis=self.quantization_config.weights_channels_axis)
