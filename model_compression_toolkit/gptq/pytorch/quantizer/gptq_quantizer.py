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
from typing import List
from model_compression_toolkit.core.common.target_platform.op_quantization_config import QuantizationMethod
from model_compression_toolkit.core.common.constants import THRESHOLD, RANGE_MAX, RANGE_MIN
from model_compression_toolkit.gptq.common.gptq_constants import THRESHOLD_TENSOR, AUXVAR
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig


class GPTQWeightQuantizer(nn.Module):

    def __init__(self,
                 weights_quantization_cfg: NodeWeightsQuantizationConfig,
                 weight_shape: torch.Size):
        """
        Construct a Base Pytorch model that utilize a fake weight quantizer
        Args:
            weights_quantization_cfg: Configuration of weight quantization.
            weight_shape: weight shape for auxiliary tensor creation.
        """
        super().__init__()

        self.signed = True
        self.num_bits = weights_quantization_cfg.weights_n_bits
        self.min_int = -int(self.signed) * (2 ** (self.num_bits - int(self.signed)))
        self.max_int = (2 ** (self.num_bits - int(self.signed))) - 1
        self.weight_shape = weight_shape
        self.trainable_params = dict()
        self.per_channel = weights_quantization_cfg.weights_per_channel_threshold
        if weights_quantization_cfg.weights_quantization_method == QuantizationMethod.UNIFORM:
            # Uniform quantization
            self.min_range = weights_quantization_cfg.weights_quantization_params.get(RANGE_MIN)
            self.max_range = weights_quantization_cfg.weights_quantization_params.get(RANGE_MAX)
        else:
            # Symmetric quantization
            threshold_values = weights_quantization_cfg.weights_quantization_params.get(THRESHOLD)
            self.min_range = -threshold_values
            self.max_range = threshold_values
        self.delta_tensor = (self.max_range - self.min_range) / (2 ** self.num_bits)


    def get_trainable_params(self) -> List:
        """
        A function to get a list of trainable parameters of the quantizer for GPTQ retraining
        Returns:
            A list of trainable tensors
        """
        return [value for value in self.trainable_params.values() if value is not None]

    def get_aux_variable(self) -> torch.Tensor:
        """
        Returns auxiliary trainable variables
        """
        return self.trainable_params.get(AUXVAR, None)


    def get_quantization_variable(self) -> torch.Tensor:
        """
        Returns quantization trainable variables
        """
        return self.trainable_params.get(THRESHOLD_TENSOR, None)

