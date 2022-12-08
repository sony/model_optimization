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

from typing import List, Any, Tuple

import torch

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder, \
    PytorchModel
from model_compression_toolkit.core.pytorch.back2framework.quantization_wrapper.quantized_layer_wrapper import \
    QuantizedLayerWrapper
from model_compression_toolkit.core.pytorch.constants import BUFFER, CONSTANT
from model_compression_toolkit.core.pytorch.reader.node_holders import BufferHolder, ConstantHolder
from model_compression_toolkit.core.pytorch.utils import get_working_device


from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.node_to_quantize_config import get_quantization_config


def get_fully_quantized_pytorch_model(graph: Graph):
    """
    Convert graph to fully quantized PyTorch model.

    Args:
        graph: Graph to convert to a PyTorch model.

    Returns:
        Fully quantized PyTorch model.
    """
    return FullyQuantizedPyTorchModelBuilder(graph=graph).build_model()



class FullyQuantizedPyTorchModel(PytorchModel):
    """
    PyTorch model with all quantization information.
    """

    def __init__(self,
                 graph: common.Graph):
        """

        Args:
            graph: Graph to build its corresponding Pytorch model.
        """

        super().__init__(graph)


    def _add_modules(self):
        """

        Add nodes in graph as modules.

        """
        for n in self.node_sort:
            if n.type == BufferHolder:
                self.add_module(n.name, node_builder(n))
                self.get_submodule(n.name).register_buffer(n.name,
                                                           torch.Tensor(n.get_weights_by_keys(BUFFER)).to(get_working_device()))
            elif n.type == ConstantHolder:
                self.add_module(n.name, node_builder(n))
                self.get_submodule(n.name).register_buffer(n.name,
                                                           torch.Tensor(n.get_weights_by_keys(CONSTANT)).to(get_working_device()))

            else:
                # Create a wrapper based on the corresponding quantization config.
                layer_wrapper = QuantizedLayerWrapper(n, get_quantization_config(n))
                # Add the wrapped layer to the model.
                self.add_module(n.name, layer_wrapper)

    def _get_op_func(self,
                     node: BaseNode,
                     configurable_nodes_names: List[str]) -> Any:
        """
        Get the operator corresponding to the passed node.

        Args:
            node: Node to get its op.
            configurable_nodes_names: List of nodes that are configurable.

        Returns:
            Operator (module) of the node.
        """
        return getattr(self, node.name)

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
        return input_tensors





class FullyQuantizedPyTorchModelBuilder(PyTorchModelBuilder):
    """
    Fully-Quantized PyTorch model.
    """

    def __init__(self,
                 graph: common.Graph):
        """

        Args:
            graph: Graph to build the model from.
        """

        super().__init__(graph)

    def build_model(self) -> Tuple[PytorchModel, UserInformation]:
        """
        Build a PyTorch fully quantized model and return it.
        Returns: Fully quantized PyTorch model and user information.

        """
        return FullyQuantizedPyTorchModel(self.graph), self.graph.user_info
