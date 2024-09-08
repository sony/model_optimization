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
import torch
from torch import nn

from mct_quantizers import mark_quantizer, QuantizationTarget, QuantizationMethod, PytorchQuantizationWrapper
from mct_quantizers.pytorch.quantizers import ActivationUniformInferableQuantizer
from model_compression_toolkit import constants as C
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerActivationConfig, TrainingMethod
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.trainable_infrastructure.common.constants import FQ_MIN, FQ_MAX
from model_compression_toolkit.trainable_infrastructure.pytorch.activation_quantizers import \
    BasePytorchActivationTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure.pytorch.quantizer_utils import uniform_quantizer


# moved (and renamed) from model_compression_toolkit/qat/pytorch/quantizer/ste_rounding/uniform_ste.py
@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.UNIFORM],
                identifier=TrainingMethod.STE)
class STEUniformActivationTrainableQuantizer(BasePytorchActivationTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize a layer activations.
    """

    def __init__(self, quantization_config: TrainableQuantizerActivationConfig, freeze_quant_params: bool = False):
        """
        Initialize a STEUniformActivationTrainableQuantizer object with parameters to use for uniform quantization.

        Args:
            quantization_config: trainable quantizer config class.
            freeze_quant_params: whether to freeze learnable quantization parameters.
        """
        super().__init__(quantization_config, freeze_quant_params)

        np_min_range = quantization_config.activation_quantization_params[C.RANGE_MIN]
        np_max_range = quantization_config.activation_quantization_params[C.RANGE_MAX]
        self.min_range_tensor = torch.Tensor([np_min_range])
        self.max_range_tensor = torch.Tensor([np_max_range])
        self.num_bits = quantization_config.activation_n_bits

    def initialize_quantization(self,
                                tensor_shape: torch.Size,
                                name: str,
                                layer: PytorchQuantizationWrapper):
        """
        Add quantizer parameters to the quantizer parameters dictionary.

        Args:
            tensor_shape: tensor shape of the quantized tensor.
            name: Tensor name.
            layer: Layer to quantize.
        """
        layer.register_parameter(name+"_"+FQ_MIN, nn.Parameter(to_torch_tensor(self.min_range_tensor),
                                                               requires_grad=not self.freeze_quant_params))
        layer.register_parameter(name+"_"+FQ_MAX, nn.Parameter(to_torch_tensor(self.max_range_tensor),
                                                               requires_grad=not self.freeze_quant_params))

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

        _min = self.get_quantizer_variable(FQ_MIN)
        _max = self.get_quantizer_variable(FQ_MAX)
        q_tensor = uniform_quantizer(inputs, _min, _max, self.num_bits)
        return q_tensor

    def convert2inferable(self) -> ActivationUniformInferableQuantizer:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            A pytorch inferable quanizer object.
        """
        _min = self.get_quantizer_variable(FQ_MIN).cpu().detach().numpy()
        _max = self.get_quantizer_variable(FQ_MAX).cpu().detach().numpy()

        return ActivationUniformInferableQuantizer(num_bits=self.num_bits,
                                                   min_range=_min.tolist(),
                                                   max_range=_max.tolist())
