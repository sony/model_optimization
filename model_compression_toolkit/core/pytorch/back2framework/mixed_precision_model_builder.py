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

from typing import List, Any, Tuple, Union, Dict

import torch
from mct_quantizers import PytorchQuantizationWrapper, PytorchActivationQuantizationHolder
from mct_quantizers.common.constants import ACTIVATION_HOLDER_QUANTIZER

from model_compression_toolkit.core import FrameworkInfo, common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder
from model_compression_toolkit.core.pytorch.mixed_precision.configurable_activation_quantizer import \
    ConfigurableActivationQuantizer
from model_compression_toolkit.core.pytorch.mixed_precision.configurable_weights_quantizer import \
    ConfigurableWeightsQuantizer
from model_compression_toolkit.logger import Logger


class MixedPrecisionPyTorchModelBuilder(PyTorchModelBuilder):
    """
    Mixed-precision PyTorch model.
    """
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

        self.graph = graph

        super().__init__(graph,
                         append2output,
                         return_float_outputs,
                         wrapper=self.mixed_precision_wrapper,
                         get_activation_quantizer_holder_fn=self.mixed_precision_activation_holder)

    def mixed_precision_wrapper(self,
                                n: common.BaseNode,
                                layer: torch.nn.Module) -> Union[PytorchQuantizationWrapper, torch.nn.Module]:
        """
        A function which takes a computational graph node and a pytorch layer and perform the quantization
        wrapping for mixed precision.

        Args:
            n: A node of mct graph.
            layer: A pytorch layer

        Returns:
            Wrapped layer with a configurable quantizer if the layer should be quantized in mixed precision, or the
            layer as is.

        Raises:
            ValueError: if kernel attribute is quantized but not configurable.
        """

        if n.kernel_attr is None or not n.is_weights_quantization_enabled(n.kernel_attr):
            return layer
        if not n.is_configurable_weight(n.kernel_attr):  # pragma: no cover
            raise ValueError(f'Weight wrapper is not expected to be created for non-configurable weight of node {n}.')
        return PytorchQuantizationWrapper(layer,
                                          weights_quantizers={
                                              n.kernel_attr: ConfigurableWeightsQuantizer(
                                                  **self._get_weights_configurable_quantizer_kwargs(n,
                                                                                                    n.kernel_attr),
                                                  kernel_attr=n.kernel_attr)})

    def _get_weights_configurable_quantizer_kwargs(self, n: BaseNode, attr: str) -> Dict[str, Any]:
        """
        Get the quantization parameters for a configurable quantizer.

        Args:
            n: The node for which the quantizer is being created.
            attr: The name of the weights attribute to be quantized.

        Returns:
            The quantization parameters as a dictionary.
        """

        assert n.candidates_quantization_cfg is not None, f"Node {n.name} candidates_quantization_cfg is None"
        node_q_cfg_candidates = n.candidates_quantization_cfg
        # sort by descending bit width so using indices would be easier
        node_q_cfg_candidates.sort(key=lambda x: (x.weights_quantization_cfg.get_attr_config(attr).weights_n_bits,
                                                  x.activation_quantization_cfg.activation_n_bits), reverse=True)

        float_weights = n.get_weights_by_keys(attr)

        max_candidate_idx = n.find_max_candidate_index()

        return {'node_q_cfg': node_q_cfg_candidates,
                'float_weights': float_weights,
                'max_candidate_idx': max_candidate_idx
                }

    def mixed_precision_activation_holder(self, n: BaseNode, holder_type: PytorchActivationQuantizationHolder = PytorchActivationQuantizationHolder) -> PytorchActivationQuantizationHolder:
        """
        Builds PytorchActivationQuantizationHolder layer with a configurable quantizer for mixed precision for a node
        with a configurable activation.

        Args:
            n: Node to get PytorchActivationQuantizationHolder to attach in its output.
            holder_type: The type of the activation quantization holder to use.

        Returns:
            A PytorchActivationQuantizationHolder layer for the node activation quantization.

        Raises:
            ValueError: if node's activation is not configurable.
        """
        if holder_type != PytorchActivationQuantizationHolder:  # pragma: no cover
            raise TypeError(f'Expected PytorchActivationQuantizationHolder holder type for mixed precision, got'
                            f'{holder_type}')

        if not n.has_configurable_activation():  # pragma: no cover
            raise ValueError(f'Activation holder is not expected to be created for a non-configurable activation of '
                             f'node {n}')

        num_of_outputs = len(n.output_shape) if isinstance(n.output_shape, list) else 1

        node_q_cfg_candidates = n.candidates_quantization_cfg

        # sorting the candidates by kernel attribute weights number of bits first and then by
        # activation number of bits (in reversed order).
        # since only kernel attribute is quantized in weights mixed precision,
        # if the node doesn't have a kernel attribute, we only sort by activation_n_bits.
        n.sort_node_candidates()

        max_candidate_idx = n.find_max_candidate_index()

        activation_quantizers = [ConfigurableActivationQuantizer(**{'node_q_cfg': node_q_cfg_candidates,
                                                                    'max_candidate_idx': max_candidate_idx,
                                                                    'kernel_attr': n.kernel_attr})] \
                                * num_of_outputs

        # Holder by definition uses a single quantizer for the activation quantization
        # thus we make sure this is the only possible case (unless it's a node with no activation
        # quantization, which in this case has an empty list).
        if len(activation_quantizers) == 1:
            return holder_type(activation_quantizers[0])

        Logger.critical(f"PytorchActivationQuantizationHolder expects a single quantizer, but ({len(activation_quantizers)}) quantizers were found for node {n}.")# pragma: no cover

    def build_model(self) -> Tuple[torch.nn.Module, UserInformation,
                                   Dict[str, Union[PytorchQuantizationWrapper, PytorchActivationQuantizationHolder]]]:
        """
        Build a PyTorch float model and return it.
        Returns: Float PyTorch model and user information.

        """
        model, user_info = super().build_model()

        # creating a mapping between graph nodes and model's layers for mixed precision configurability
        model_layers = dict(model.named_children())
        conf_node2layers = {n.name: self._find_layers_in_model_by_node(n, model_layers)
                            for n in self.graph.get_configurable_sorted_nodes()}

        return model, user_info, conf_node2layers


    @staticmethod
    def _get_weights_quant_layers(n: BaseNode, named_layers: Dict[str, torch.nn.Module]) \
            -> List[PytorchQuantizationWrapper]:
        """
        Filters PytorchQuantizationWrapper layers from an MP model that are matching to the given graph node.

        Args:
            n: A configurable graph node.
            named_layers: Mixed precision model layers list.

        Returns: A list of layers that responsible for the node's weights quantization.

        """
        return [module for m_name, module in named_layers.items() if isinstance(module, PytorchQuantizationWrapper)
                and m_name == n.name]

    @staticmethod
    def _get_activation_quant_layers(n: BaseNode, named_layers: Dict[str, torch.nn.Module]) \
            -> List[PytorchActivationQuantizationHolder]:
        """
        Filters PytorchActivationQuantizationHolder layers from an MP model that are matching to the given graph node.

        Args:
            n: A configurable graph node.
            named_layers: Mixed precision model layers list.

        Returns: A list of layers that responsible for the node's activation quantization.

        """
        return [module for m_name, module in named_layers.items()
                if isinstance(module, PytorchActivationQuantizationHolder) and
                m_name.replace(f"_{ACTIVATION_HOLDER_QUANTIZER}", '') == n.name]

    def _find_layers_in_model_by_node(self, n: BaseNode, named_layers: Dict[str, torch.nn.Module]) -> \
            List[Union[PytorchQuantizationWrapper, PytorchActivationQuantizationHolder]]:
        """
        Retries layers from an MP model that are matching to the given graph node, that is, this are either
        PytorchQuantizationWrapper layers or PytorchActivationQuantizationHolder layers that are responsible for the graph
        configurable model quantization.

        Args:
            n: A configurable graph node.
            named_layers: Mixed precision model layers list.

        Returns: A list of layers that responsible for the node's quantization.

        """
        # Only layers with kernel op are considered weights configurable
        weights_quant = False if n.kernel_attr is None else n.is_weights_quantization_enabled(n.kernel_attr)
        act_quant = n.is_activation_quantization_enabled()

        if weights_quant and not act_quant:
            return self._get_weights_quant_layers(n, named_layers)
        elif not weights_quant and act_quant:
            return self._get_activation_quant_layers(n, named_layers)
        elif weights_quant and act_quant:
            return self._get_weights_quant_layers(n, named_layers) + self._get_activation_quant_layers(n, named_layers)
        else:  # pragma: no cover
            Logger.critical(f"Expected node {n.name} to have either weights or activation quantization enabled, but both are disabled.")
