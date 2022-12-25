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
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.gptq.pytorch.quantizer.gumbel_rounding.base_gumbel_weights_quantizer import BaseGumbelWeightQuantizer, init_aux_var
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from model_compression_toolkit.gptq.pytorch.quantizer.quant_utils import symmetric_quantizer
from model_compression_toolkit.gptq.pytorch.quantizer.quant_utils import ste_clip, ste_gumbel, gumbel_softmax, power_of_two_max
from model_compression_toolkit.gptq.common.gptq_constants import AUXVAR, THRESHOLD_TENSOR, TEMP, SCALE_TENSOR
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig
from model_compression_toolkit.core.common.constants import THRESHOLD


class SymmetricGumbelWeightQuantizer(BaseGumbelWeightQuantizer):
    """
    Class that implements a quantizer with trainable parameters to be used for GPTQ training.
    """

    def __init__(self,
                 weights_quantization_cfg: NodeWeightsQuantizationConfig,
                 gptq_config: GradientPTQConfig,
                 weight: torch.Tensor):
        """
        Construct a Pytorch model that utilize a fake weight quantizer of Symmetric Gumbel rounding
        Args:
            weights_quantization_cfg: Configuration of weight quantization
            gptq_config: GradientPTQConfig object with parameters about the tuning process.
            weight: weight for auxiliary tensor creation.
        """
        super().__init__(weights_quantization_cfg, gptq_config, weight.shape)
        self.signed = True
        self.min_int = -int(self.signed) * (2 ** (self.num_bits - int(self.signed)))
        self.max_int = (2 ** (self.num_bits - int(self.signed))) - 1
        self.threshold_tensor = to_torch_tensor(weights_quantization_cfg.weights_quantization_params.get(THRESHOLD))
        self.scale_tensor = torch.ones(self.weight_shape)

        # Set trainable tensors
        self.set_trainable_params(weight)


    def set_trainable_params(self, weight: torch.nn.Parameter):
        """
        A function to set a list of trainable parameters of the quantizer for GPTQ retraining
        Args:
            weight: weight for auxiliary tensor creation.
        """
        q_error = weight - symmetric_quantizer(weight,
                                               self.threshold_tensor,
                                               num_bits=self.num_bits,
                                               signed=True,
                                               power_of_two=self.power_of_two)
        ceil_indicator = (q_error < 0).int()  # Negative error means the choosen point is rounded to ceil.
        self.aux_tensor = nn.Parameter(to_torch_tensor(init_aux_var(ceil_indicator, self.weight_shape, self.m)), requires_grad=True)
        self.trainable_params.update({AUXVAR: self.aux_tensor})
        self.temp_tensor = nn.Parameter(to_torch_tensor(self.maximal_temp*torch.ones([1,*self.weight_shape])), requires_grad=True)
        self.trainable_params.update({TEMP: self.temp_tensor})
        if self.quantization_parameter_learning and not self.power_of_two:
            self.scale_tensor = nn.Parameter(to_torch_tensor(self.scale_tensor), requires_grad=True)
            self.trainable_params.update({SCALE_TENSOR: self.scale_tensor})
        elif self.quantization_parameter_learning:
            self.threshold_tensor = nn.Parameter(self.threshold_tensor, requires_grad=True)
            self.trainable_params.update({THRESHOLD_TENSOR: self.threshold_tensor})
        else:
            self.trainable_params.update({THRESHOLD_TENSOR: self.threshold_tensor})

    def get_aux_variable(self) -> torch.Tensor:
        """
        Returns auxiliary trainable variables
        """
        return self.trainable_params.get(AUXVAR)

    def get_quantization_variable(self) -> Union[torch.Tensor, List]:
        """
        Returns quantization trainable variables
        """
        if self.quantization_parameter_learning and not self.power_of_two:
            return [self.trainable_params.get(SCALE_TENSOR)]
        else:
            return [self.trainable_params.get(THRESHOLD_TENSOR)]

    def get_temperature_variable(self) -> Union[torch.Tensor, List]:
        """
        Returns temperature trainable variables
        """
        return self.trainable_params.get(TEMP)

    def get_weight_quant_params(self) -> dict:
        """
        Returns weight quantization dictionary params
        """
        threshold_tensor = self.threshold_tensor
        if self.power_of_two:
            threshold_tensor = power_of_two_max(threshold_tensor)
        elif self.quantization_parameter_learning:
            threshold_tensor = threshold_tensor*self.scale_tensor
        return {THRESHOLD: torch_tensor_to_numpy(threshold_tensor.detach())}

    def forward(self, w: nn.Parameter, training:bool = True) -> nn.Parameter:
        """
        Weight fake quantizer
        Args:
            w: weights to quantize.
            training: whether in training mode or not
        Returns:
            quantized weights
        """
        self.update_iteration(training)

        #####################################################
        # Gumbel Softmax
        #####################################################
        if training:
            gumbel_scale = self.gumbel_scale if self.gumbel_scale_per_bitwidth is None \
                else self.gumbel_scale_per_bitwidth.get(self.num_bits, self.gumbel_scale)
            self.p_t = gumbel_softmax(self.aux_tensor, self.tau, self.g_t, gumbel_scale=gumbel_scale)
        else:
            self.p_t = ste_gumbel(gumbel_softmax(self.aux_tensor, self.minimal_temp, 0))

        auxhat_tensor = torch.sum(self.p_t * self.shift_tensor.reshape(self.reshape_aux_shift), dim=0)

        #####################################################
        # Quantizer
        #####################################################
        threshold_tensor = self.threshold_tensor
        if self.power_of_two:
            threshold_tensor = power_of_two_max(threshold_tensor)
        delta_tensor = threshold_tensor / (2 ** (self.num_bits-int(self.signed)))
        w0 = torch.floor(w / delta_tensor).detach()
        w1 = w0 + auxhat_tensor
        w2 = ste_clip(w1, min_val=self.min_int, max_val=self.max_int)
        w_q = delta_tensor * w2
        # Scale
        if self.quantization_parameter_learning and not self.power_of_two:
            w_q *= self.scale_tensor
        return w_q

