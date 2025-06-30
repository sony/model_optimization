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

from typing import List, Tuple

import torch

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.quantization.quantization_fn_selection import get_activation_quantization_fn
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder, \
    PytorchModel


class QuantizedPyTorchModel(PytorchModel):
    """
    Quantized PyTorch model.
    """

    def __init__(self,
                 graph: common.Graph,
                 append2output=None):
        """

        Args:
            graph: Graph to build its corresponding Pytorch model.
            append2output: List of nodes or OutTensor objects.
        """

        super().__init__(graph,
                         append2output)

    def _quantize_node_activations(self,
                                   node: BaseNode,
                                   input_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Quantize node's activation given input tensors.

        Args:
            node: Node to quantize its outputs.
            input_tensors: Input tensors of the node.

        Returns:
            Output of the node.

        """
        if node.is_activation_quantization_enabled():
            if isinstance(input_tensors, list):
                input_tensors = torch.cat(input_tensors, dim=0)
            activation_quantizer = get_activation_quantization_fn(node.final_activation_quantization_cfg)
            return activation_quantizer(input_tensors)
        return input_tensors


class QuantizedPyTorchModelBuilder(PyTorchModelBuilder):

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 return_float_outputs: bool = False):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes to append to model's output.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        super().__init__(graph,
                         append2output,
                         return_float_outputs)

    def build_model(self) -> Tuple[PytorchModel, UserInformation]:
        """
        Build a PyTorch quantized model and return it.
        Returns: Quantized PyTorch model and user information.

        """
        return QuantizedPyTorchModel(self.graph,
                                     self.append2output), self.graph.user_info
