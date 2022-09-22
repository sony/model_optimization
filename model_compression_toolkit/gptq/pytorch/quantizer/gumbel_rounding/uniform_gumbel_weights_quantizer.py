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
from typing import List, Union, Tuple, Dict
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.gptq.pytorch.quantizer.gumbel_rounding.base_gumbel_weights_quantizer import BaseGumbelWeightQuantizer, init_aux_var
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from model_compression_toolkit.gptq.pytorch.quantizer.quant_utils import ste_clip, ste_gumbel, gumbel_softmax
from model_compression_toolkit.gptq.common.gptq_constants import AUXVAR, PTQ_MAX_RANGE, PTQ_MIN_RANGE, TEMP
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig
from model_compression_toolkit.core.common.constants import RANGE_MAX, RANGE_MIN

class UniformGumbelWeightQuantizer(BaseGumbelWeightQuantizer):
    """
    Class that implements a quantizer with trainable parameters to be used for GPTQ training.
    """

    def __init__(self,
                 weights_quantization_cfg: NodeWeightsQuantizationConfig,
                 gptq_config: GradientPTQConfig,
                 weight_shape: torch.Size):
        """
        Construct a Pytorch model that utilize a fake weight quantizer of Uniform Gumbel rounding
        Args:
            weights_quantization_cfg: Configuration of weight quantization
            gptq_config: GradientPTQConfig object with parameters about the tuning process.
            weight_shape: weight shape for auxiliary tensor creation.
        """
        super().__init__(weights_quantization_cfg, gptq_config, weight_shape)
        self.min_int = 0
        self.max_int = 2**self.num_bits - 1
        self.max_range_tensor = weights_quantization_cfg.weights_quantization_params.get(RANGE_MAX)
        self.min_range_tensor = weights_quantization_cfg.weights_quantization_params.get(RANGE_MIN)

        # Set trainable tensors
        self.set_trainable_params()


    def set_trainable_params(self):
        """
        A function to set a list of trainable parameters of the quantizer for GPTQ retraining
        """
        self.aux_tensor = nn.Parameter(to_torch_tensor(init_aux_var(self.weight_shape, self.m)), requires_grad=True)
        self.trainable_params.update({AUXVAR: self.aux_tensor})
        self.temp_tensor = nn.Parameter(to_torch_tensor(self.maximal_temp*torch.ones([1,*self.weight_shape])), requires_grad=True)
        self.trainable_params.update({TEMP: self.temp_tensor})
        self.max_range_tensor = nn.Parameter(to_torch_tensor(self.max_range_tensor), requires_grad=self.quantization_parameter_learning)
        self.trainable_params.update({PTQ_MAX_RANGE: self.max_range_tensor})
        self.min_range_tensor = nn.Parameter(to_torch_tensor(self.min_range_tensor), requires_grad=self.quantization_parameter_learning)
        self.trainable_params.update({PTQ_MIN_RANGE: self.min_range_tensor})

    def get_aux_variable(self) -> torch.Tensor:
        """
        Returns auxiliary trainable variables
        """
        return self.trainable_params.get(AUXVAR)

    def get_quantization_variable(self) -> Union[torch.Tensor, List]:
        """
        Returns quantization trainable variables
        """
        return [self.trainable_params.get(PTQ_MAX_RANGE), self.trainable_params.get(PTQ_MIN_RANGE)]

    def get_temperature_variable(self) -> Union[torch.Tensor, List]:
        """
        Returns temperature trainable variables
        """
        return self.trainable_params.get(TEMP)

    def get_weight_quant_params(self) -> Dict[str, np.ndarray]:
        """
        Returns weight quantization dictionary params
        """
        max_range_tensor = self.max_range_tensor
        min_range_tensor = self.min_range_tensor
        return {PTQ_MAX_RANGE: torch_tensor_to_numpy(max_range_tensor.detach()),
                PTQ_MIN_RANGE: torch_tensor_to_numpy(min_range_tensor.detach())}

    @staticmethod
    def fix_range_to_include_zero(range_min: torch.Tensor,
                                  range_max: torch.Tensor,
                                  n_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjusting the quantization range to include representation of 0.0 in the quantization grid.
        If quantization per-channel, then range_min and range_max should be tensors in the specific shape that allows
        quantization along the channel_axis.

        Args:
            range_min: min bound of the quantization range (before adjustment).
            range_max: max bound of the quantization range (before adjustment).
            n_bits: Number of bits to quantize the tensor.

        Returns: adjusted quantization range
        """
        min_positive = range_min > 0
        max_negative = range_max < 0
        mid_range = torch.logical_and(torch.logical_not(min_positive), torch.logical_not(max_negative))
        min_positive = min_positive.float()
        max_negative = max_negative.float()
        mid_range = mid_range.float()

        scale = (range_max - range_min) / (2 ** n_bits - 1)
        min_range_adj = scale * torch.round(range_min / scale)
        max_range_adj = range_max - range_min + min_range_adj

        min_range_adj = min_range_adj * mid_range + max_negative * range_min
        max_range_adj = max_range_adj * mid_range + min_positive * range_max
        return min_range_adj, max_range_adj

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
            self.p_t = gumbel_softmax(self.aux_tensor, self.tau, self.g_t)
        else:
            self.p_t = ste_gumbel(gumbel_softmax(self.aux_tensor, self.minimal_temp, 0))

        auxhat_tensor = torch.sum(self.p_t * self.shift_tensor.reshape(self.reshape_aux_shift), dim=0)

        #####################################################
        # Quantizer
        #####################################################
        max_range_tensor = self.max_range_tensor
        min_range_tensor = self.min_range_tensor

        # adjusts the quantization rage so the quantization grid include zero.
        a, b = self.fix_range_to_include_zero(min_range_tensor, max_range_tensor, self.num_bits)

        # Compute the step size of quantized values.
        delta_tensor = (b - a) / (2 ** self.num_bits - 1)

        # Apply rounding
        w0 = torch.round((w - a) / delta_tensor).detach()  # Apply rounding

        w1 = w0 + auxhat_tensor

        # Clip data in range
        w2 = ste_clip(w1, min_val=self.min_int, max_val=self.max_int)

        # Quantize the data between min/max of quantization range.
        w_q = delta_tensor * w2 + a
        return w_q

