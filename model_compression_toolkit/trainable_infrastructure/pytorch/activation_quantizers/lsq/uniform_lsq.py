# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from torch import nn

from mct_quantizers import mark_quantizer, QuantizationTarget, QuantizationMethod, PytorchQuantizationWrapper
from mct_quantizers.pytorch.quantizers import ActivationUniformInferableQuantizer
from model_compression_toolkit import constants as C
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import fix_range_to_include_zero
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerActivationConfig, TrainingMethod
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.trainable_infrastructure.common.constants import FQ_MIN, FQ_MAX
from model_compression_toolkit.trainable_infrastructure.pytorch.activation_quantizers import \
    BasePytorchActivationTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure.pytorch.quantizer_utils import uniform_lsq_quantizer


# moved (and renamed) from model_compression_toolkit/qat/pytorch/quantizer/lsq/uniform_lsq.py
@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.UNIFORM],
                identifier=TrainingMethod.LSQ)
class LSQUniformActivationTrainableQuantizer(BasePytorchActivationTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize layer activations.
    """

    def __init__(self, quantization_config: TrainableQuantizerActivationConfig):
        """
        Initialize a LSQUniformActivationTrainableQuantizer object with parameters to use
        for uniform quantization.

        Args:
            quantization_config: trainable quantizer config class
        """
        super().__init__(quantization_config)
        self.num_bits = self.quantization_config.activation_n_bits
        self.min_int = 0
        self.max_int = 2 ** self.num_bits - 1
        self.min_range = np.array([quantization_config.activation_quantization_params[C.RANGE_MIN]])
        self.max_range = np.array([quantization_config.activation_quantization_params[C.RANGE_MAX]])

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
        layer.register_parameter(name+"_"+FQ_MIN, nn.Parameter(to_torch_tensor(self.min_range), requires_grad=True))
        layer.register_parameter(name+"_"+FQ_MAX, nn.Parameter(to_torch_tensor(self.max_range), requires_grad=True))

        # Save the quantizer parameters for later calculations
        self.add_quantizer_variable(FQ_MIN, layer.get_parameter(name+"_"+FQ_MIN), VariableGroup.QPARAMS)
        self.add_quantizer_variable(FQ_MAX, layer.get_parameter(name+"_"+FQ_MAX), VariableGroup.QPARAMS)

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
        min_range = self.get_quantizer_variable(FQ_MIN)
        max_range = self.get_quantizer_variable(FQ_MAX)
        n_channels = inputs.shape[1]
        scale_factor = 1.0 / np.sqrt(self.max_int * n_channels)
        inputs_quantized = uniform_lsq_quantizer(inputs, min_range, max_range, self.num_bits, self.min_int, self.max_int, scale_factor)
        return inputs_quantized

    def convert2inferable(self) -> ActivationUniformInferableQuantizer:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            A pytorch inferable quanizer object.
        """
        min_range = self.get_quantizer_variable(FQ_MIN).cpu().detach().numpy()
        max_range = self.get_quantizer_variable(FQ_MAX).cpu().detach().numpy()
        min_range, max_range = fix_range_to_include_zero(min_range, max_range, self.num_bits)
        return ActivationUniformInferableQuantizer(num_bits=self.num_bits,
                                                   min_range=min_range.tolist(),
                                                   max_range=max_range.tolist())
