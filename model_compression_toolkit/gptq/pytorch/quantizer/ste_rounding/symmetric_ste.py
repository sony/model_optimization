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
import torch
import torch.nn as nn
from typing import List, Union
import numpy as np

from model_compression_toolkit import quantizers_infrastructure as qi
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.gptq.common.gptq_config import RoundingType
from model_compression_toolkit.gptq.pytorch.quantizer.base_pytorch_gptq_quantizer import \
    BasePytorchGPTQTrainableQuantizer
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.gptq.pytorch.quantizer.quant_utils import ste_round, ste_clip
from model_compression_toolkit.gptq.common.gptq_constants import AUXVAR
from model_compression_toolkit.core.common.constants import THRESHOLD
from model_compression_toolkit.quantizers_infrastructure import TrainableQuantizerWeightsConfig
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer


@mark_quantizer(quantization_target=qi.QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                quantizer_type=RoundingType.STE)
class STEWeightQuantizer(BasePytorchGPTQTrainableQuantizer):
    """
    Trainable symmetric quantizer to quantize a layer weights.
    """

    def __init__(self,
                 quantization_config: TrainableQuantizerWeightsConfig):
        """
        Construct a Pytorch model that utilize a fake weight quantizer of STE (Straight Through Estimator) for symmetric quantizer.

        Args:
            quantization_config: Trainable weights quantizer config.
        """
        super().__init__(quantization_config)

        self.signed = True
        self.num_bits = quantization_config.weights_n_bits
        self.per_channel = quantization_config.weights_per_channel_threshold

        threshold_values = quantization_config.weights_quantization_params[THRESHOLD]
        self.threshold_shape = np.asarray(threshold_values).shape
        self.threshold_values = np.reshape(np.asarray(threshold_values), [-1]) if self.per_channel else float(
            threshold_values)

        self.quantization_axis = quantization_config.weights_channels_axis
        self.power_of_two = quantization_config.weights_quantization_method == QuantizationMethod.POWER_OF_TWO

        # Set trainable tensors
        # self.set_trainable_params()

        # Create tensors
        # self.delta_tensor = self.threshold_values / (2 ** (self.num_bits - int(self.signed)))
        # self.max_delta_change = gptq_config.lsb_change_per_bit_width.get(self.num_bits)
        # self.delta_tensor = to_torch_tensor(self.delta_tensor)
        # self.max_tensor_change = self.delta_tensor * self.max_delta_change

        # self.min_int = -int(self.signed) * (2 ** (self.num_bits - int(self.signed)))
        # self.max_int = (2 ** (self.num_bits - int(self.signed))) - 1

    def set_trainable_params(self):
        """
        A function to set a list of trainable parameters of the quantizer for GPTQ retraining
        """
        self.aux_tensor = nn.Parameter(to_torch_tensor(torch.zeros(self.weight_shape)), requires_grad=True)
        self.trainable_params.update({AUXVAR: self.aux_tensor})

    def get_aux_variable(self) -> torch.Tensor:
        """
        Returns auxiliary trainable variables
        """
        return self.trainable_params.get(AUXVAR)

    def get_quantization_variable(self) -> Union[torch.Tensor, List]:
        """
        Returns quantization trainable variables
        """
        return []

    def get_weight_quantization_params(self) -> dict:
        """
        Returns weight quantization dictionary params
        """
        return {THRESHOLD: self.threshold_values}

    def forward(self, w: nn.Parameter, training: bool = True) -> nn.Parameter:
        """
        Weight fake quantizer
        Args:
            w: weights to quantize.
            training: whether in training mode or not
        Returns:
            quantized weights
        """
        v0 = ste_clip(self.aux_tensor, min_val=-self.max_tensor_change, max_val=self.max_tensor_change)
        v1 = v0 / self.delta_tensor
        w0 = torch.round(w / self.delta_tensor).detach()
        w1 = w0 + v1
        w2 = ste_round(w1)
        w3 = ste_clip(w2, min_val=self.min_int, max_val=self.max_int)
        w_q = self.delta_tensor * w3
        return w_q

