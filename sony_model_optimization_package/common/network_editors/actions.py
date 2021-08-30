# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================

from abc import ABC, abstractmethod
from collections import namedtuple

from sony_model_optimization_package.common.graph.node import Node
from sony_model_optimization_package.common.quantization.quantization_params_fn_selection import \
    get_activation_quantization_params_fn, get_weights_quantization_params_fn

_EditRule = namedtuple('EditRule', 'filter action')


class EditRule(_EditRule):
    """
    A tuple of a node filter and an action. The filter matches nodes in the graph which represents the model,
    and the action is applied on these nodes during the quantization process.

    Examples:
        Create an EditRule to quantize all Conv2D wights using 9 bits:

        >>> import sony_model_optimization_package as smop
        >>> from tensorflow.keras.layers import Conv2D
        >>> er_list = [EditRule(filter=smop.network_editor.NodeTypeFilter(Conv2D),
        >>> action=smop.network_editor.ChangeWeightsQuantConfigAttr(weights_n_bits=9))]

        Then the rules list can be passed to :func:`~sony_model_optimization_package.keras_post_training_quantization`
        to modify the network during the quantization process.

    """

    pass


class BaseAction(ABC):
    """
    Base class for actions. An action class applies a defined action on a node.
    """

    @abstractmethod
    def apply(self, node: Node, graph, fw_info):
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


class ChangeWeightsQuantConfigAttr(BaseAction):
    """
    Class ChangeQuantConfigAttr to change attributes in a node's weights quantization configuration.
    """

    def __init__(self, **kwargs):
        """
        Init a ChangeWeightsQuantConfigAttr object.

        Args:
            kwargs: dict of attr_name and attr_value to change in the node's weights quantization configuration.
        """
        self.kwargs = kwargs

    def apply(self, node: Node, graph, fw_info):
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
        if node.weights_quantization_cfg is not None:
            for attr_name, attr_value in self.kwargs.items():
                node.weights_quantization_cfg.set_quant_config_attr(attr_name, attr_value)


class ChangeActivationQuantConfigAttr(BaseAction):
    """
    Class ChangeQuantConfigAttr to change attributes in a node's activation quantization configuration.
    """

    def __init__(self, **kwargs):
        """
        Init a ChangeActivationQuantConfigAttr object.

        Args:
            kwargs: dict of attr_name and attr_value to change in the node's activation quantization configuration.
        """
        self.kwargs = kwargs

    def apply(self, node: Node, graph, fw_info):
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

    def apply(self, node: Node, graph, fw_info):
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
            node.weights_quantization_cfg.set_weights_quantization_params_fn(self.weights_quantization_params_fn)


class ChangeQuantizationMethod(BaseAction):
    """
    Class ChangeQuantizationMethod to change a node's weights/activations quantizer function.
    """

    def __init__(self, activation_quantization_method=None, weights_quantization_method=None):
        """
        Init a ChangeQuantizationParamFunction object.

        Args:
            activation_quantization_method: a quantization method for a node's activations.
            weights_quantization_method: a quantization method for a node's weights.
        """
        self.activation_quantization_method = activation_quantization_method
        self.weights_quantization_method = weights_quantization_method

    def apply(self, node: Node, graph, fw_info):
        """
        Change the node's weights/activations quantization function.

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

        if self.weights_quantization_method is not None:

            weights_quantization_params_fn = get_weights_quantization_params_fn(self.weights_quantization_method,
                                                                                node.weights_quantization_cfg.weights_threshold_method)

            node.weights_quantization_cfg.set_weights_quantization_params_fn(weights_quantization_params_fn)

            weights_quantization_fn = fw_info.weights_quantizer_mapping.get(self.weights_quantization_method)

            if weights_quantization_fn is None:
                raise Exception('Unknown quantization method for weights')

            node.weights_quantization_cfg.set_weights_quantization_fn(weights_quantization_fn)
