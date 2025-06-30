# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Callable

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.graph.base_node import BaseNode


_EditRule = namedtuple('EditRule', 'filter action')


class EditRule(_EditRule):
    """
    A tuple of a node filter and an action. The filter matches nodes in the graph which represents the model,
    and the action is applied on these nodes during the quantization process.

    Examples:
        Create an EditRule to quantize all Conv2D kernel attribute weights using 9 bits:

        >>> import model_compression_toolkit as mct
        >>> from model_compression_toolkit.core.keras.constants import KERNEL
        >>> from tensorflow.keras.layers import Conv2D
        >>> er_list = [mct.core.network_editor.EditRule(filter=mct.core.network_editor.NodeTypeFilter(Conv2D), action=mct.core.network_editor.ChangeCandidatesWeightsQuantConfigAttr(attr_name=KERNEL, weights_n_bits=9))]

        Then the rules list can be passed to :func:`~model_compression_toolkit.keras_post_training_quantization`
        to modify the network during the quantization process.

    """

    def __repr__(self):
        _str = f'filter={type(self.filter).__name__}{self.filter.__dict__}'  # pragma: no cover
        _str = f'{_str} ; action={type(self.action).__name__}{self.action.kwargs}'  # pragma: no cover
        return _str  # pragma: no cover

    pass


class BaseAction(ABC):
    """
    Base class for actions. An action class applies a defined action on a node.
    """

    @abstractmethod
    def apply(self, node: BaseNode, graph):
        """
        Apply an action on the node after matching the node with a node filter.

        Args:
            node: Node to apply the action on.
            graph: Graph to apply the action on.

        Returns:
            Node after action is applied.

        """
        pass  # pragma: no cover


class ChangeCandidatesWeightsQuantConfigAttr(BaseAction):
    """
    Change attributes in a layer's weights quantization configuration candidates.
    """

    def __init__(self, attr_name: str = None, **kwargs):
        """
        Args:
            attr_name: The weights attribute's name to set the weights quantization params function for.
            kwargs: Dictionary of attr_name and attr_value to change layer's weights quantization configuration candidates.
        """
        self.kwargs = kwargs
        self.attr_name = attr_name

    def apply(self, node: BaseNode, graph):
        """
        Change the attribute 'attr_name' in weights quantization config candidates with 'attr_value'.

        Args:
            node: Node object to change its quant_config.
            graph: Graph to apply the action on.
        Returns:
            The node after its weights' quantization config candidates have been modified.
        """

        for nqc in node.candidates_quantization_cfg:
            for parameter_name, parameter_value in self.kwargs.items():
                nqc.weights_quantization_cfg.set_quant_config_attr(parameter_name, parameter_value,
                                                                   attr_name=self.attr_name)


class ChangeFinalWeightsQuantConfigAttr(BaseAction):
    """
    Change attributes in a layer's final weights quantization config.
    """

    def __init__(self, attr_name: str = None, **kwargs):
        """
        Args:
            attr_name: The weights attribute's name to set the weights quantization params function for.
            kwargs: Dictionary of attr_name and attr_value to change layer's final weights quantization config.
        """
        self.kwargs = kwargs
        self.attr_name = attr_name

    def apply(self, node: BaseNode, graph):
        if node.final_weights_quantization_cfg is not None:
            for parameter_name, parameter_value in self.kwargs.items():
                node.final_weights_quantization_cfg.set_quant_config_attr(parameter_name, parameter_value,
                                                                          self.attr_name)


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

    def apply(self, node: BaseNode, graph):
        """
        Change the attribute 'attr_name' in activation quantization configuration candidates with 'attr_value'.

        Args:
            node: Node object to change its quant_config.
            graph: Graph to apply the action on.
        """
        for nqc in node.candidates_quantization_cfg:
            for parameter_name, parameter_value in self.kwargs.items():
                nqc.activation_quantization_cfg.set_quant_config_attr(parameter_name, parameter_value)


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

    def apply(self, node: BaseNode, graph):
        if node.final_activation_quantization_cfg is not None:
            for parameter_name, parameter_value in self.kwargs.items():
                node.final_activation_quantization_cfg.set_quant_config_attr(parameter_name, parameter_value)


class ChangeFinalActivationQuantizationMethod(BaseAction):
    """
    Class ChangeFinalActivationQuantizationMethod to change a node's weights/activations quantizer function.
    """

    def __init__(self, activation_quantization_method=None):
        """
        Init a ChangeFinalActivationQuantizationMethod object.

        Args:
            activation_quantization_method: a quantization method for a node's activations.
        """

        self.activation_quantization_method = activation_quantization_method

    def apply(self, node: BaseNode, graph):
        """
        Change the node's activations quantization function.

        Args:
            node: Node object to change its threshold selection function.
            graph: Graph to apply the action on.

        Returns:
            The node after its quantization function has been modified.
        """

        if self.activation_quantization_method is not None and node.final_activation_quantization_cfg is not None:
            node.final_activation_quantization_cfg.activation_quantization_method = self.activation_quantization_method


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

    def apply(self, node: BaseNode, graph):
        """
        Change the node's activations quantization function.

        Args:
            node: Node object to change its threshold selection function.
            graph: Graph to apply the action on.

        """
        if self.activation_quantization_method is not None:
            for qc in node.candidates_quantization_cfg:
                qc.activation_quantization_cfg.activation_quantization_method = self.activation_quantization_method


class ChangeFinalWeightsQuantizationMethod(BaseAction):
    """
    Class ChangeFinalWeightsQuantizationMethod to change a node's weights/activations quantizer method.
    """

    def __init__(self, attr_name: str, weights_quantization_method=None):
        """
        Init a ChangeFinalWeightsQuantizationMethod object.

        Args:
            attr_name: The weights attribute's name to set the weights quantization method for.
            weights_quantization_method: a quantization method for a node's weights.
        """

        self.weights_quantization_method = weights_quantization_method
        self.attr_name = attr_name

    def apply(self, node: BaseNode, graph):
        """
        Change the node's weights quantization function.

        Args:
            node: Node object to change its threshold selection function.
            graph: Graph to apply the action on.

        """

        if self.weights_quantization_method is not None and node.final_weights_quantization_cfg is not None:
            attr_config = node.final_weights_quantization_cfg.get_attr_config(self.attr_name)
            attr_config.weights_quantization_method = self.weights_quantization_method


class ChangeCandidatesWeightsQuantizationMethod(BaseAction):
    """
    Class ChangeCandidatesWeightsQuantizationMethod to change a node's weights quantizer function.
    """

    def __init__(self, attr_name: str, weights_quantization_method: QuantizationMethod = None):
        """
        Init a ChangeCandidatesWeightsQuantizationMethod object.

        Args:
            weights_quantization_method: a quantization method for a node's weights.
            attr_name: The weights attribute's name to set the weights quantization params function for.
        """
        self.weights_quantization_method = weights_quantization_method
        self.attr_name = attr_name

    def apply(self, node: BaseNode, graph: Graph):
        """
        Change the node's weights quantization function.

        Args:
            node: Node object to change its threshold selection function.
            graph: Graph to apply the action on.

        Returns:
            The node after its quantization function has been modified.
        """

        if self.weights_quantization_method is not None:
            for qc in node.candidates_quantization_cfg:
                attr_qc = qc.weights_quantization_cfg.get_attr_config(self.attr_name)
                attr_qc.weights_quantization_method = self.weights_quantization_method


class ReplaceLayer(BaseAction):

    def __init__(self, layer_type: type, get_params_and_weights_fn: Callable):
        """

        Args:
            layer_type: node's layer type to replace with existing one
            get_params_and_weights_fn: function that modifies the layer's params and weights.
                                        The function receives two arguments, a dictionary of weights and layer's
                                        framework attributes and returns new weights and framework attributes to use.

        """
        self.layer_type = layer_type
        self.get_params_and_weights_fn = get_params_and_weights_fn

    def apply(self, node: BaseNode, graph: Graph):
        """
        Replacing node's layer type and configurations

        Args:
            node: Node object to replace or modify
            graph: Graph to apply the action on.

        Returns:
            The node after its layer functionality has been modified.
        """
        activation_quantization_cfg = {}
        if node.final_activation_quantization_cfg is not None:
            activation_quantization_cfg = node.final_activation_quantization_cfg.activation_quantization_params

        weights, config = self.get_params_and_weights_fn(node.weights, activation_quantization_cfg,
                                                         **node.framework_attr)
        node.framework_attr = config
        node.weights = weights
        node.layer_class = self.layer_type
        Logger.warning(f'Layer {node.name} was replaced but quantization parameters were set by original layer')
