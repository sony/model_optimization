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
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.gptq.pytorch.quantizer.gptq_quantizer import GPTQWeightQuantizer
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.gptq.pytorch.quantizer.quant_utils import ste_round, ste_clip
from model_compression_toolkit.gptq.common.gptq_constants import AUXVAR
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig


class STEWeightQuantizer(GPTQWeightQuantizer):
    """
    Class that wraps a Pytorch layer (nn.Module) with trainable parameters to be used for GPTQ training.
    """

    def __init__(self,
                 weights_quantization_cfg: NodeWeightsQuantizationConfig,
                 gptq_config: GradientPTQConfig,
                 weight_shape: torch.Size):
        """
        Construct a Pytorch model that utilize a fake weight quantizer of STE (Straight Through Estimator)
        Args:
            weights_quantization_cfg: Configuration of weight quantization
            gptq_config: GradientPTQConfig object with parameters about the tuning process.
            weight_shape: weight shape for auxiliary tensor creation.
        """
        super().__init__(weights_quantization_cfg=weights_quantization_cfg, weight_shape=weight_shape)

        self.max_delta_change = gptq_config.lsb_change_per_bit_width.get(self.num_bits)
        self.train_bias = gptq_config.train_bias

        # Set trainable tensors
        self.set_trainable_params()

        # Create tensors
        self.delta_tensor = to_torch_tensor(self.delta_tensor)
        self.max_tensor_change = self.delta_tensor * self.max_delta_change

    def set_trainable_params(self):
        """
        A function to set a list of trainable parameters of the quantizer for GPTQ retraining
        """
        self.aux_tensor = nn.Parameter(to_torch_tensor(torch.zeros(self.weight_shape)), requires_grad=True)
        self.trainable_params.update({AUXVAR: self.aux_tensor})

    def forward(self, w: nn.Parameter) -> nn.Parameter:
        """
        Weight fake quantizer
        Args:
            w: weights to quantize.
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

