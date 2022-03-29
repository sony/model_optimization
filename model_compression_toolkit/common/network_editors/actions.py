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
from model_compression_toolkit.common.quantization import quantization_params_generation
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

    def __repr__(self):
        _str = f'filter={type(self.filter).__name__}{self.filter.__dict__}'
        _str = f'{_str} ; action={type(self.action).__name__}{self.action.kwargs}'
        return _str

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
    Change attributes in a layer's weights quantization configuration candidates.
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: Dictionary of attr_name and attr_value to change layer's weights quantization configuration candidates.
        """
        self.kwargs = kwargs

    def apply(self, node: BaseNode, graph, fw_info):
        """
        Change the attribute 'attr_name' in weights quantization config candidates with 'attr_value'.

        Args:
            node: Node object to change its quant_config.
            graph: Graph to apply the action on.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
                     groups of layers by how they should be quantized, etc.)
        Returns:
            The node after its weights' quantization config candidates have been modified.
        """
        for nqc in node.candidates_quantization_cfg:
            for attr_name, attr_value in self.kwargs.items():
                nqc.weights_quantization_cfg.set_quant_config_attr(attr_name, attr_value)


class ChangeFinalWeightsQuantConfigAttr(BaseAction):
    """
    Change attributes in a layer's final weights quantization config.
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: Dictionary of attr_name and attr_value to change layer's final weights quantization config.
        """
        self.kwargs = kwargs

    def apply(self, node: BaseNode, graph, fw_info):
        if node.final_weights_quantization_cfg is not None:
            for attr_name, attr_value in self.kwargs.items():
                node.final_weights_quantization_cfg.set_quant_config_attr(attr_name, attr_value)


class ChangeCandidatesActivationQuantConfigAttr(BaseAction):
    """
    Change attributes in a layer's activation quantization configuration candidates.
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: Dictionary of attr_name and attr_value to change in the layer's activation quantization configuration candidates.
        """
        self.kwargs = kwargs

    def apply(self, node: BaseNode, graph, fw_info):
        """
        Change the attribute 'attr_name' in activation quantization configuration candidates with 'attr_value'.

        Args:
            node: Node object to change its quant_config.
            graph: Graph to apply the action on.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
                     groups of layers by how they should be quantized, etc.)
        Returns:q
            The node after its activation quantization configuration candidates have been modified.
        """
        for nqc in node.candidates_quantization_cfg:
            for attr_name, attr_value in self.kwargs.items():
                nqc.activation_quantization_cfg.set_quant_config_attr(attr_name, attr_value)


class ChangeFinalActivationQuantConfigAttr(BaseAction):
    """
    Change attributes in a layer's final activation quantization config.
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs: Dictionary of attr_name and attr_value to change layer's final activation quantization config.
        """
        self.kwargs = kwargs

    def apply(self, node: BaseNode, graph, fw_info):
        if node.final_activation_quantization_cfg is not None:
            for attr_name, attr_value in self.kwargs.items():
                node.final_activation_quantization_cfg.set_quant_config_attr(attr_name, attr_value)


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
        for nqc in node.candidates_quantization_cfg:
            if self.activation_quantization_params_fn is not None:
                nqc.activation_quantization_cfg.set_activation_quantization_params_fn(
                    self.activation_quantization_params_fn)
            if self.weights_quantization_params_fn is not None:
                nqc.weights_quantization_cfg.set_weights_quantization_params_fn(self.weights_quantization_params_fn)


class ChangeCandidatesActivationQuantizationMethod(BaseAction):
    """
    Class ChangeQuantizationMethod to change a node's activations quantizer function.
    """

    def __init__(self, activation_quantization_method=None):
        """
        Init a ChangeCandidatesActivationQuantizationMethod object.

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
            for qc in node.candidates_quantization_cfg:
                activation_quantization_params_fn = get_activation_quantization_params_fn(
                    self.activation_quantization_method, qc.activation_quantization_cfg.activation_error_method)

                if node.prior_info.is_output_bounded():
                    activation_quantization_params_fn = quantization_params_generation.no_clipping_selection_min_max

                qc.activation_quantization_cfg.set_activation_quantization_params_fn(activation_quantization_params_fn)
                activation_quantization_fn = fw_info.activation_quantizer_mapping.get(self.activation_quantization_method)

                if activation_quantization_fn is None:
                    raise Exception('Unknown quantization method for activations')

                qc.activation_quantization_cfg.set_activation_quantization_fn(activation_quantization_fn)


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


class ChangeCandidatesWeightsQuantizationMethod(BaseAction):
    """
    Class ChangeCandidatesWeightsQuantizationMethod to change a node's weights quantizer function.
    """

    def __init__(self, weights_quantization_method=None):
        """
        Init a ChangeCandidatesWeightsQuantizationMethod object.

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
            for qc in node.candidates_quantization_cfg:

                weights_quantization_params_fn = get_weights_quantization_params_fn(self.weights_quantization_method,
                                                                                    qc.weights_quantization_cfg.weights_error_method)

                qc.weights_quantization_cfg.set_weights_quantization_params_fn(weights_quantization_params_fn)

                weights_quantization_fn = fw_info.weights_quantizer_mapping.get(self.weights_quantization_method)

                if weights_quantization_fn is None:
                    raise Exception('Unknown quantization method for weights')

                qc.weights_quantization_cfg.set_weights_quantization_fn(weights_quantization_fn)
