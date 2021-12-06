# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

from abc import ABC, abstractmethod
from collections import namedtuple

from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.common.quantization.quantization_params_fn_selection import \
    get_activation_quantization_params_fn, get_weights_quantization_params_fn

_EditRule = namedtuple('EditRule', 'filter action')


class EditRule(_EditRule):
    """
    A tuple of a node filter and an action. The filter matches nodes in the graph which represents the model,
    and the action is applied on these nodes during the quantization process.

    Examples:
        Create an EditRule to quantize all Conv2D wights using 9 bits:

        >>> import model_compression_toolkit as mct
        >>> from tensorflow.keras.layers import Conv2D
        >>> er_list = [EditRule(filter=mct.network_editor.NodeTypeFilter(Conv2D),
        >>> action=mct.network_editor.ChangeCandidatesWeightsQuantConfigAttr(weights_n_bits=9))]

        Then the rules list can be passed to :func:`~model_compression_toolkit.keras_post_training_quantization`
        to modify the network during the quantization process.

    """

    pass


class BaseAction(ABC):
    """
    Base class for actions. An action class applies a defined action on a node.
    """

    @abstractmethod
    def apply(self, node: BaseNode, graph, fw_info):
        """
        Apply an action on the node after matching the node with a node filter.

        Args:
            node: Node to apply the action on.
            graph: Graph to apply the action on.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
                     groups of layers by how they should be quantized, etc.)

        Returns:
            Node after action is applied.

        """
        pass


class ChangeCandidatesWeightsQuantConfigAttr(BaseAction):
    """
    Class ChangeCandidatesWeightsQuantConfigAttr to change attributes in a node's weights quantization configuration.
    """

    def __init__(self, **kwargs):
        """
        Init a ChangeCandidatesWeightsQuantConfigAttr object.

        Args:
            kwargs: dict of attr_name and attr_value to change in the node's weights quantization configuration.
        """
        self.kwargs = kwargs

    def apply(self, node: BaseNode, graph, fw_info):
        """
        Change the attribute 'attr_name' in quant_config with 'attr_value'.

        Args:
            node: Node object to change its quant_config.
            graph: Graph to apply the action on.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
                     groups of layers by how they should be quantized, etc.)
        Returns:
            The node after its quant_config has been modified.
        """
        if node.candidates_weights_quantization_cfg is not None:
            for nqc in node.candidates_weights_quantization_cfg:
                for attr_name, attr_value in self.kwargs.items():
                    nqc.set_quant_config_attr(attr_name, attr_value)


class ChangeFinalWeightsQuantConfigAttr(BaseAction):
    """
    Class ChangeFinalWeightsQuantConfigAttr to change attributes in a node's quant_config.
    """

    def __init__(self, **kwargs):
        """
        Init a ChangeFinalWeightsQuantConfigAttr object.

        Args:
            kwargs: dict of attr_name and attr_value to change in the node's quant_config.
        """
        self.kwargs = kwargs

    def apply(self, node: BaseNode, graph, fw_info):
        if node.final_weights_quantization_cfg is not None:
            for attr_name, attr_value in self.kwargs.items():
                node.final_weights_quantization_cfg.set_quant_config_attr(attr_name, attr_value)



class ChangeActivationQuantConfigAttr(BaseAction):
    """
    Class ChangeActivationQuantConfigAttr to change attributes in a node's activation quantization configuration.
    """

    def __init__(self, **kwargs):
        """
        Init a ChangeActivationQuantConfigAttr object.

        Args:
            kwargs: dict of attr_name and attr_value to change in the node's activation quantization configuration.
        """
        self.kwargs = kwargs

    def apply(self, node: BaseNode, graph, fw_info):
        """
        Change the attribute 'attr_name' in quant_config with 'attr_value'.

        Args:
            node: Node object to change its quant_config.
            graph: Graph to apply the action on.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
                     groups of layers by how they should be quantized, etc.)
        Returns:q
            The node after its quant_config has been modified.
        """
        if node.activation_quantization_cfg is not None:
            for attr_name, attr_value in self.kwargs.items():
                node.activation_quantization_cfg.set_quant_config_attr(attr_name, attr_value)


class ChangeQuantizationParamFunction(BaseAction):
    """
    Class ChangeQuantizationParamFunction to change a node's weights/activations quantization params function.
    """

    def __init__(self, activation_quantization_params_fn=None, weights_quantization_params_fn=None):
        """
        Init a ChangeQuantizationParamFunction object.

        Args:
            activation_quantization_params_fn: a params function for a node's activations.
            weights_quantization_params_fn: a params function for a node's weights.
        """
        self.activation_quantization_params_fn = activation_quantization_params_fn
        self.weights_quantization_params_fn = weights_quantization_params_fn

    def apply(self, node: BaseNode, graph, fw_info):
        """
        Change the node's weights/activations quantization params function.

        Args:
            node: Node object to change its quantization params function.
            graph: Graph to apply the action on.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
                     groups of layers by how they should be quantized, etc.)

        Returns:
            The node after its quantization params function has been modified.
        """
        if self.activation_quantization_params_fn is not None:
            node.activation_quantization_cfg.set_activation_quantization_params_fn(
                self.activation_quantization_params_fn)
        if self.weights_quantization_params_fn is not None:
            for qc in node.candidates_weights_quantization_cfg:
                qc.set_weights_quantization_params_fn(self.weights_quantization_params_fn)


class ChangeActivationQuantizationMethod(BaseAction):
    """
    Class ChangeQuantizationMethod to change a node's activations quantizer function.
    """

    def __init__(self, activation_quantization_method=None):
        """
        Init a ChangeActivationQuantizationMethod object.

        Args:
            activation_quantization_method: a quantization method for a node's activations.
        """
        self.activation_quantization_method = activation_quantization_method

    def apply(self, node: BaseNode, graph, fw_info):
        """
        Change the node's activations quantization function.

        Args:
            node: Node object to change its threshold selection function.
            graph: Graph to apply the action on.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
                     groups of layers by how they should be quantized, etc.)

        Returns:
            The node after its quantization function has been modified.
        """
        if self.activation_quantization_method is not None:

            out_stats_container = graph.get_out_stats_collector(node)[0] if isinstance(
                graph.get_out_stats_collector(node),
                list) else graph.get_out_stats_collector(
                node)

            activation_quantization_params_fn = get_activation_quantization_params_fn(
                self.activation_quantization_method,
                node.activation_quantization_cfg.activation_threshold_method,
                out_stats_container.use_min_max)

            node.activation_quantization_cfg.set_activation_quantization_params_fn(activation_quantization_params_fn)
            activation_quantization_fn = fw_info.activation_quantizer_mapping.get(self.activation_quantization_method)

            if activation_quantization_fn is None:
                raise Exception('Unknown quantization method for activations')

            node.activation_quantization_cfg.set_activation_quantization_fn(activation_quantization_fn)



class ChangeFinalWeightsQuantizationMethod(BaseAction):
    """
    Class ChangeFinalWeightsQuantizationMethod to change a node's weights/activations quantizer function.
    """

    def __init__(self, weights_quantization_method=None):
        """
        Init a ChangeFinalWeightsQuantizationMethod object.

        Args:
            weights_quantization_method: a quantization method for a node's weights.
        """

        self.weights_quantization_method = weights_quantization_method

    def apply(self, node: BaseNode, graph, fw_info):
        """
        Change the node's weights quantization function.

        Args:
            node: Node object to change its threshold selection function.
            graph: Graph to apply the action on.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
                     groups of layers by how they should be quantized, etc.)

        Returns:
            The node after its quantization function has been modified.
        """

        if self.weights_quantization_method is not None:

            weights_quantization_params_fn = get_weights_quantization_params_fn(self.weights_quantization_method,
                                                                                node.final_weights_quantization_cfg.weights_threshold_method)

            node.final_weights_quantization_cfg.set_weights_quantization_params_fn(weights_quantization_params_fn)

            weights_quantization_fn = fw_info.weights_quantizer_mapping.get(self.weights_quantization_method)

            if weights_quantization_fn is None:
                raise Exception('Unknown quantization method for weights')

            node.final_weights_quantization_cfg.set_weights_quantization_fn(weights_quantization_fn)



class ChangeCandidtaesWeightsQuantizationMethod(BaseAction):
    """
    Class ChangeCandidtaesWeightsQuantizationMethod to change a node's weights quantizer function.
    """

    def __init__(self, weights_quantization_method=None):
        """
        Init a ChangeCandidtaesWeightsQuantizationMethod object.

        Args:
            weights_quantization_method: a quantization method for a node's weights.
        """
        self.weights_quantization_method = weights_quantization_method

    def apply(self, node: BaseNode, graph, fw_info):
        """
        Change the node's weights quantization function.

        Args:
            node: Node object to change its threshold selection function.
            graph: Graph to apply the action on.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
                     groups of layers by how they should be quantized, etc.)

        Returns:
            The node after its quantization function has been modified.
        """

        if self.weights_quantization_method is not None:
            for qc in node.candidates_weights_quantization_cfg:

                weights_quantization_params_fn = get_weights_quantization_params_fn(self.weights_quantization_method,
                                                                                    qc.weights_threshold_method)

                qc.set_weights_quantization_params_fn(weights_quantization_params_fn)

                weights_quantization_fn = fw_info.weights_quantizer_mapping.get(self.weights_quantization_method)

                if weights_quantization_fn is None:
                    raise Exception('Unknown quantization method for weights')

                qc.set_weights_quantization_fn(weights_quantization_fn)
