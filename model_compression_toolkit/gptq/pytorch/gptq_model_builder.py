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
from typing import Tuple, List
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import BaseNode
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder, PytorchModel
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.gptq.pytorch.quantizer.quantizer_wrapper import quantizer_wrapper


class GPTQPytorchModel(PytorchModel):
    """
    Class for GPTQ PyTorch model.
    """

    def __init__(self,
                 graph: common.Graph,
                 gptq_config: GradientPTQConfig,
                 append2output=None,
                 return_float_outputs: bool = True):
        """
        Args:
            graph: Graph to build the model from.
            gptq_config: Configuration for GPTQ optimization.
            append2output: Nodes to append to model's output.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        super().__init__(graph,
                         append2output,
                         DEFAULT_PYTORCH_INFO,
                         return_float_outputs)

        for node in graph.nodes():
            if not isinstance(node, FunctionalNode):
                self.add_module(node.name, quantizer_wrapper(node, gptq_config))

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
        return node.final_activation_quantization_cfg.quantize_node_output(input_tensors)


class GPTQPytorchModelBuilder(PyTorchModelBuilder):
    """
    Builder of GPTQ Pytorch models.
    """

    def __init__(self,
                 graph: common.Graph,
                 gptq_config: GradientPTQConfig,
                 append2output=None,
                 return_float_outputs: bool = True):
        """

        Args:
            graph: Graph to build the model from.
            gptq_config: Configuration for GPTQ optimization.
            append2output: Nodes to append to model's output.
            return_float_outputs: Whether the model returns float tensors or not.
        """
        super().__init__(graph,
                         append2output,
                         DEFAULT_PYTORCH_INFO,
                         return_float_outputs)
        self.gptq_config = gptq_config

    def build_model(self) -> Tuple[PytorchModel, UserInformation]:
        """
        Build a GPTQ PyTorch model and return it.
        Returns:
            GPTQ PyTorch model and user information.
        """
        return GPTQPytorchModel(self.graph,
                                self.gptq_config,
                                self.append2output,
                                self.return_float_outputs), self.graph.user_info
