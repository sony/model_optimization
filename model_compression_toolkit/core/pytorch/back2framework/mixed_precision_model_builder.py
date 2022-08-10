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

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.pytorch.back2framework.instance_builder import node_builder
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder, \
    PytorchModel

from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.mixed_precision.mixed_precision_wrapper import PytorchMixedPrecisionWrapper


class MixedPrecisionPyTorchModel(PytorchModel):

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO):
        """

        Args:
            graph: Graph to build its corresponding Pytorch model.
            append2output: List of nodes or OutTensor objects.
            fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
        """

        super().__init__(graph,
                         append2output,
                         fw_info)


    def _add_modules(self):
        configurable_nodes = self.graph.get_configurable_sorted_nodes()
        for n in self.node_sort:
            if n in configurable_nodes:
                self.add_module(n.name, PytorchMixedPrecisionWrapper(n, self.fw_info))
            else:
                if not isinstance(n, FunctionalNode):
                    self.add_module(n.name, node_builder(n))

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
        if node.is_all_activation_candidates_equal():
            # otherwise, we want to use the float tensor when building the model for MP search
            if isinstance(input_tensors, list):
                input_tensors = torch.cat(input_tensors, dim=0)
            input_tensors = node.candidates_quantization_cfg[0].activation_quantization_cfg.quantize_node_output(input_tensors)
        return input_tensors


    def _get_op_func(self,
                     node: BaseNode,
                     configurable_nodes_names: List[str]) -> Any:
        """
        Gets the operation function that runs the actual inference of the nodes compatible layer.

        Args:
            node: The corresponding node of the layer it runs.
            configurable_nodes_names: A list of names of configurable nodes in the quantized model.

        Returns: Module/functional to apply to the input tensors.

        """
        if node.name in configurable_nodes_names:
            return getattr(self, node.name)
        else:
            return node.type if isinstance(node, FunctionalNode) else getattr(self, node.name)




class MixedPrecisionPyTorchModelBuilder(PyTorchModelBuilder):
    """
    Mixed-precision PyTorch model.
    """
    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                 return_float_outputs: bool = False):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes to append to model's output.
            fw_info: Information about the specific framework of the model that is built.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs)

    def build_model(self) -> Tuple[PytorchModel, UserInformation]:
        """
        Build a PyTorch float model and return it.
        Returns: Float PyTorch model and user information.

        """
        return MixedPrecisionPyTorchModel(self.graph,
                                          self.append2output,
                                          self.fw_info), self.graph.user_info
