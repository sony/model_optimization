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
from model_compression_toolkit.core.common import BaseNode, Logger
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig, RoundingType
from model_compression_toolkit.gptq.pytorch.quantizer.gptq_quantizer import BaseWeightQuantizer
from model_compression_toolkit.gptq.pytorch.quantizer.ste_rounding.ste_weights_quantizer import STEWeightQuantizer
from model_compression_toolkit.core.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.core.pytorch.constants import KERNEL
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor


class WeightQuantizerWrapper(nn.Module):

    def __init__(self, node: BaseNode, gptq_config: GradientPTQConfig, weight_quantizer: BaseWeightQuantizer):
        """
        Construct a Pytorch model that constitutes as a wrapper for a Pytorch layer, built from a given graph node.
        Args:
            node: Node to build its Pytorch quantizer wrapper.
            gptq_config: GradientPTQConfig object with parameters about the tuning process.
            weight_quantizer: BaseWeightQuantizer object for gradient based weight quantizer
        """
        super().__init__()

        # loading operation
        self.op = node.type(**node.framework_attr)

        # loading the weights from the graph node (weights of the trained model)
        self.op.load_state_dict({k: torch.Tensor(v) for k, v in node.weights.items()}, strict=False)
        self.float_weight = to_torch_tensor(getattr(self.op, KERNEL)).detach()

        # replace non-gradient needed nn.Parameter with gradient needed torch.tensor
        delattr(self.op, KERNEL)
        setattr(self.op, KERNEL, self.float_weight)
        setattr(getattr(self.op, KERNEL), 'requires_grad', True)

        # quantizer
        self.weight_quantizer = weight_quantizer(node.final_weights_quantization_cfg, gptq_config, self.float_weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Weight fake quantizer wrapper
        Args:
            x: input to layer.
        Returns:
            Output of layer after using operation with fake quantized weights
        """
        # Run weight quantizer
        setattr(self.op, KERNEL, self.weight_quantizer(self.float_weight))
        # Do computation
        return self.op(x)


def quantizer_wrapper(node: BaseNode, gptq_config: GradientPTQConfig) -> nn.Module:
    """
    Construct a Pytorch model that constitutes as a wrapper for a Pytorch layer, built from a given graph node.
    Args:
        node: Node to build its Pytorch layer.
        gptq_config: GradientPTQConfig with parameters about the tuning process.
    """
    if node.is_weights_quantization_enabled():
        if gptq_config.rounding_type == RoundingType.STE:
            node_instance = WeightQuantizerWrapper(node, gptq_config, STEWeightQuantizer)
        else:
            Logger.critical('No support for GumbelRounding GPTQ yet. Work in progress..')
    else:
        # No quantization
        node_instance = node_builder(node)

    return node_instance
