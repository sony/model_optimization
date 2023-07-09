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
import torch
import torch.nn as nn
from typing import Dict
import numpy as np
from model_compression_toolkit.core.common.defaultdict import DefaultDict

from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget, PytorchQuantizationWrapper
from model_compression_toolkit.gptq.common.gptq_config import RoundingType
from model_compression_toolkit.gptq.pytorch.quantizer.base_pytorch_gptq_quantizer import \
    BasePytorchGPTQTrainableQuantizer
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from model_compression_toolkit.gptq.pytorch.quantizer import quant_utils as qutils
from model_compression_toolkit.gptq.common.gptq_constants import AUXVAR, PTQ_THRESHOLD, MAX_LSB_CHANGE
from model_compression_toolkit.constants import THRESHOLD
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig
from mct_quantizers import mark_quantizer
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.trainable_infrastructure.common.quant_utils import \
    get_threshold_reshape_shape


def pertubation_symmetric_quantizer(input_tensor: torch.Tensor,
                                    auxvar_tensor: nn.Parameter,
                                    max_tensor: torch.Tensor,
                                    num_bits: int,
                                    signed: bool,
                                    power_of_two: bool,
                                    max_lsbs_change: int = MAX_LSB_CHANGE) -> nn.Parameter:
    """
    Quantize a tensor symmetrically with maximum LSBs shift.

    Args:
        input_tensor: Tensor to quantize. values of this tensor are not changed during gptq.
        auxvar_tensor: Tensor that manifests the bit shift the weight due to gptq
        max_tensor: Tensor with max values to compute the threshold.
        num_bits: Num of bits to use.
        signed: Signedness of the quantization range.
        power_of_two: Whether the threshold should be constrained or not.
        max_lsbs_change: maximum number of LSBs that the auxvar is allowed to change

    Returns:
        A quantized tensor.
    """

    if power_of_two:
        max_tensor = qutils.power_of_two_max(max_tensor)
    delta = qutils.calculate_delta(max_tensor, num_bits, signed)
    delta = to_torch_tensor(delta)
    max_tensor_change = delta * max_lsbs_change

    min_int = -int(signed) * (2 ** (num_bits - int(signed)))
    max_int = (2 ** (num_bits - int(signed))) - 1

    tensor_clipped = qutils.ste_clip(auxvar_tensor, min_val=-max_tensor_change, max_val=max_tensor_change) / delta
    input_tensor_int = torch.round(input_tensor / delta).detach()

    tensor_q = qutils.ste_round(qutils.ste_round(input_tensor_int + tensor_clipped))

    return delta * qutils.ste_clip(tensor_q, max_val=max_int, min_val=min_int)


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=RoundingType.STE)
class STEWeightGPTQQuantizer(BasePytorchGPTQTrainableQuantizer):
    """
    Trainable symmetric quantizer to quantize a layer weights.
    """

    def __init__(self,
                 quantization_config: TrainableQuantizerWeightsConfig,
                 max_lsbs_change_map: dict = DefaultDict({}, lambda: 1)):
        """
        Construct a Pytorch model that utilize a fake weight quantizer of STE (Straight Through Estimator) for symmetric quantizer.

        Args:
            quantization_config: Trainable weights quantizer config.
        """
        super().__init__(quantization_config)
        self.num_bits = quantization_config.weights_n_bits
        self.per_channel = quantization_config.weights_per_channel_threshold

        threshold_values = quantization_config.weights_quantization_params[THRESHOLD]
        self.threshold_shape = np.asarray(threshold_values).shape
        self.threshold_values = np.reshape(np.asarray(threshold_values), [-1]) if self.per_channel else float(
            threshold_values)

        self.quantization_axis = quantization_config.weights_channels_axis
        self.power_of_two = quantization_config.weights_quantization_method == QuantizationMethod.POWER_OF_TWO
        self.max_lsbs_change = max_lsbs_change_map.get(self.num_bits)


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

        layer.register_parameter(f"{name}_{PTQ_THRESHOLD}",
                                 nn.Parameter(torch.tensor(self.threshold_values, requires_grad=False)
                                              if not self.per_channel
                                              else to_torch_tensor(self.threshold_values),requires_grad=False))
        layer.register_parameter(f"{name}_{AUXVAR}", nn.Parameter(to_torch_tensor(torch.zeros(self.threshold_shape)),
                                                                  requires_grad=True))

        # save the quantizer added parameters for later calculations
        self.add_quantizer_variable(PTQ_THRESHOLD, layer.get_parameter(f"{name}_{PTQ_THRESHOLD}"), VariableGroup.QPARAMS)
        self.add_quantizer_variable(AUXVAR, layer.get_parameter(f"{name}_{AUXVAR}"), VariableGroup.WEIGHTS)


    def get_quant_config(self) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        old_threshold = self.get_quantizer_variable(PTQ_THRESHOLD)
        return {THRESHOLD: torch_tensor_to_numpy(old_threshold).reshape(self.threshold_shape)}

    def __call__(self,
                 inputs: nn.Parameter,
                 training: bool) -> nn.Parameter:
        """
        Quantize a tensor.

        Args:
            inputs: Input tensor to quantize.
            training: whether in training mode or not

        Returns:
            quantized tensor
        """
        auxvar = self.get_quantizer_variable(AUXVAR)
        ptq_threshold_tensor = self.get_quantizer_variable(PTQ_THRESHOLD)

        if self.per_channel:
            reshape_shape = get_threshold_reshape_shape(inputs.shape,
                                                        quant_axis=self.quantization_axis,
                                                        quant_axis_dim=-1)
            ptq_threshold_tensor = torch.reshape(ptq_threshold_tensor, reshape_shape)

            q_tensor = pertubation_symmetric_quantizer(inputs,
                                                       auxvar,
                                                       ptq_threshold_tensor,
                                                       self.num_bits,
                                                       signed=True,
                                                       power_of_two=self.power_of_two,
                                                       max_lsbs_change=self.max_lsbs_change)
            return q_tensor
        else:
            return pertubation_symmetric_quantizer(inputs,
                                                   auxvar,
                                                   ptq_threshold_tensor,
                                                   self.num_bits,
                                                   signed=True,
                                                   power_of_two=self.power_of_two)
