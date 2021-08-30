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
from collections import Callable
from typing import Dict, Any
from sony_model_optimization_package.common.quantization.quantization_config import QuantizationMethod
from sony_model_optimization_package.common.defaultdict import DefaultDict
from sony_model_optimization_package.common.graph.node import Node


class FrameworkInfo(object):

    def __init__(self,
                 kernel_ops: list,
                 activation_ops: list,
                 no_quantization_ops: list,
                 activation_quantizer_mapping: Dict[QuantizationMethod, Callable],
                 weights_quantizer_mapping: Dict[QuantizationMethod, Callable],
                 kernel_channels_mapping: DefaultDict,
                 activation_min_max_mapping: Dict[str, tuple],
                 layer_min_max_mapping: Dict[Any, tuple]):
        """
        A class to wrap all information about a specific framework the library needs to quantize a model.
        Specifically, FrameworkInfo holds lists of layers by how they should be quantized, and multiple mappings such as
        layer to it kernel channels indices, and a layer to its min/max values, etc.
        The layers lists are divided into three groups:
        kernel_ops: Layers that have coefficients and need to get quantized (e.g., Conv2D, Dense, etc.)
        activation_ops: Layers that their outputs should get quantized (e.g., Add, ReLU, etc.)
        no_quantization_ops:Layers that should not get quantized (e.g., Reshape, Transpose, etc.)

        Args:
            kernel_ops (list): A list of operators that are in the kernel_ops group.
            activation_ops (list): A list of operators that are in the activation_ops group.
            no_quantization_ops (list): A list of operators that are in the no_quantization_ops group.
            activation_quantizer_mapping (Dict[QuantizationMethod, Callable]): A dictionary mapping from QuantizationMethod to a quantization function.
            weights_quantizer_mapping (Dict[QuantizationMethod, Callable]): A dictionary mapping from QuantizationMethod to a quantization function.
            kernel_channels_mapping (DefaultDict): Dictionary from a layer to a tuple of its kernel in/out channels indices.
            activation_min_max_mapping (Dict[str, tuple]): Dictionary from an activation function to its min/max output values.
            layer_min_max_mapping (Dict[Any, tuple]): Dictionary from a layer to its min/max output values.

        Examples:
            When quantizing a Keras model, if we want to quantize the kernels of Conv2D layers only, we can
            set, and we know it's kernel out/in channel indices are (3, 2) respectivly:

            >>> import tensorflow as tf
            >>> kernel_ops = [tf.keras.layers.Conv2D]
            >>> kernel_channels_mapping = DefaultDict({tf.keras.layers.Conv2D: (3,2)})

            Then, we can create a FrameworkInfo object:

            >>> FrameworkInfo(kernel_ops, [], [], kernel_channels_mapping, {}, {})

            and pass it to :func:`~sony_model_optimization_package.keras_post_training_quantization`.

            To quantize the activations of ReLU, we can create a new FrameworkInfo instance:

            >>> activation_ops = [tf.keras.layers.ReLU]
            >>> FrameworkInfo(kernel_ops, activation_ops, [], kernel_channels_mapping, {}, {})

            If we don't want to quantize a layer (e.g. Reshape), we can add it to the no_no_quantization_ops list:

            >>> no_quantization_ops = [tf.keras.layers.Reshape]
            >>> FrameworkInfo(kernel_ops, activation_ops, no_quantization_ops, kernel_channels_mapping, {}, {})

            If an activation layer (tf.keras.layers.Activation) should be quantized and we know it's min/max outputs range in advanced, we can add it to activation_min_max_mapping for saving the statistics collection time. For example:

            >>> activation_min_max_mapping = {'softmax': (0, 1)}
            >>> FrameworkInfo(kernel_ops, activation_ops, no_quantization_ops, kernel_channels_mapping, activation_min_max_mapping, {})

            If a layer's activations should be quantized and we know it's min/max outputs range in advanced, we can add it to layer_min_max_mapping for saving the statistics collection time. For example:

            >>> layer_min_max_mapping = {tf.keras.layers.Softmax: (0, 1)}
            >>> FrameworkInfo(kernel_ops, activation_ops, no_quantization_ops, kernel_channels_mapping, activation_min_max_mapping, layer_min_max_mapping)

        """

        self.kernel_ops = kernel_ops
        self.activation_ops = activation_ops
        self.no_quantization_ops = no_quantization_ops
        self.activation_quantizer_mapping = activation_quantizer_mapping
        self.weights_quantizer_mapping = weights_quantizer_mapping
        self.kernel_channels_mapping = kernel_channels_mapping
        self.activation_min_max_mapping = activation_min_max_mapping
        self.layer_min_max_mapping = layer_min_max_mapping


    def layers_has_min_max(self, layer: Any) -> bool:
        """
        Check if a layer is in a layer to min/max mapping the FrameworkInfo holds.
        Args:
            layer: A layer to check if has a min/max known values.

        Returns:
            Whether a layer has a min/max known values or not.
        """

        return layer in self.layer_min_max_mapping

    def in_kernel_ops(self, n: Node) -> bool:
        """
        Check whether a node is in the kernel_ops group or not.

        Args:
            n: A node to check.

        Returns:
            Whether the node is in the kernel_ops group or not.
        """

        return n.layer_class in self.kernel_ops

    def in_activation_ops(self, n: Node) -> bool:
        """
        Check whether a node is in the activation group or not.

        Args:
            n: A node to check.

        Returns:
            Whether the node is in the activation group or not.
        """

        return n.layer_class in self.activation_ops

    def in_no_quantization_ops(self, n: Node) -> bool:
        """
        Check whether a node is in the no quantization group or not.

        Args:
            n: A node to check.

        Returns:
            Whether the node is in the no quantization group or not.
        """

        return n.layer_class in self.no_quantization_ops
