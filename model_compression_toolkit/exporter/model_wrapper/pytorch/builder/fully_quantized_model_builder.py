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

from typing import Union, Callable
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.constants import FOUND_TORCH
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common import BaseNode

if FOUND_TORCH:
    import torch
    from mct_quantizers import PytorchQuantizationWrapper, PytorchActivationQuantizationHolder
    from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder
    from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.node_to_quantizers import \
        get_quantization_quantizers


    def fully_quantized_wrapper(node: common.BaseNode,
                                module: torch.nn.Module) -> Union[torch.nn.Module,PytorchQuantizationWrapper]:
        """
        A function which takes a computational graph node and a pytorch module and
        perform the quantization wrapping

        Args:
            node: A node of mct graph.
            module: A Pytorch module
        Returns: Wrapped layer

        """
        weight_quantizers, _ = get_quantization_quantizers(node)
        if len(weight_quantizers) > 0:
            return PytorchQuantizationWrapper(module, weight_quantizers)
        return module

    def get_activation_quantizer_holder(node: BaseNode) -> Callable:
        """
        Retrieve a PytorchActivationQuantizationHolder layer to use for activation quantization of a node.
        If the layer is not supposed to be wrapped with an activation quantizer - return None.
        Args:
            node: Node to attach a PytorchActivationQuantizationHolder to its output.
        Returns:
            A PytorchActivationQuantizationHolder module for the node's activation quantization.
        """
        _, activation_quantizers = get_quantization_quantizers(node)
        # Holder by definition uses a single quantizer for the activation quantization
        # thus we make sure this is the only possible case (unless it's a node we no activation
        # quantization, which in this case has an empty list).
        if len(activation_quantizers) == 1:
            return PytorchActivationQuantizationHolder(activation_quantizers[0])
        Logger.error(
            f'PytorchActivationQuantizationHolder supports a single quantizer but {len(activation_quantizers)} quantizers '
            f'were found for node {node}')

    def get_exportable_pytorch_model(graph: Graph):
        """
        Convert graph to fully quantized PyTorch model.

        Args:
            graph: Graph to convert to a PyTorch model.

        Returns:
            Fully quantized PyTorch model.
        """
        return PyTorchModelBuilder(graph=graph,
                                   wrapper=fully_quantized_wrapper,
                                   get_activation_quantizer_holder_fn=get_activation_quantizer_holder).build_model()

else:
    def get_exportable_pytorch_model(*args, **kwargs):  # pragma: no cover
        Logger.error('Installing torch is mandatory '
                     'when using get_exportable_pytorch_model. '
                     'Could not find PyTorch package.')