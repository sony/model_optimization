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


from collections.abc import Callable
from enum import Enum
from typing import Dict, Any, List



from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod


class ChannelAxis(Enum):
    """

    Index of output channels axis:

    NHWC - Output channels index is last.

    NCHW - Output channels index is 1.

    """
    NHWC = -1
    NCHW = 1


class FrameworkInfo:

    def __init__(self,
                 activation_quantizer_mapping: Dict[QuantizationMethod, Callable],
                 kernel_channels_mapping: DefaultDict,
                 activation_min_max_mapping: Dict[str, tuple],
                 layer_min_max_mapping: Dict[Any, tuple],
                 kernel_ops_attributes_mapping: DefaultDict,
                 out_channel_axis_mapping: DefaultDict):
        """
        A class to wrap all information about a specific framework the library needs to quantize a model.
        Specifically, FrameworkInfo holds lists of layers by how they should be quantized, and multiple mappings such as
        layer to it kernel channels indices, and a layer to its min/max values, etc.
        The layers lists are divided into three groups:
        kernel_ops: Layers that have coefficients and need to get quantized (e.g., Conv2D, Dense, etc.)
        activation_ops: Layers that their outputs should get quantized (e.g., Add, ReLU, etc.)
        no_quantization_ops:Layers that should not get quantized (e.g., Reshape, Transpose, etc.)

        Args:
            activation_quantizer_mapping (Dict[QuantizationMethod, Callable]): A dictionary mapping from QuantizationMethod to a quantization function.
            kernel_channels_mapping (DefaultDict): Dictionary from a layer to a tuple of its kernel in/out channels indices.
            activation_min_max_mapping (Dict[str, tuple]): Dictionary from an activation function to its min/max output values.
            layer_min_max_mapping (Dict[Any, tuple]): Dictionary from a layer to its min/max output values.
            kernel_ops_attributes_mapping (DefaultDict): Dictionary from a framework operator to a list of its weights attirbutes to quantize.
            out_channel_axis_mapping (DefaultDict): Dictionary of output channels of the model's layers (for computing statistics per-channel).

        Examples:
            When quantizing a Keras model, if we want to quantize the kernels of Conv2D layers only, we can
            set, and we know it's kernel out/in channel indices are (3, 2) respectivly:

            >>> import tensorflow as tf
            >>> kernel_ops = [tf.keras.layers.Conv2D]
            >>> kernel_channels_mapping = DefaultDict({tf.keras.layers.Conv2D: (3,2)})

            Then, we can create a FrameworkInfo object:

            >>> FrameworkInfo(kernel_channels_mapping, {}, {})

            If an activation layer (tf.keras.layers.Activation) should be quantized and we know it's min/max outputs range in advanced, we can add it to activation_min_max_mapping for saving the statistics collection time. For example:

            >>> activation_min_max_mapping = {'softmax': (0, 1)}
            >>> FrameworkInfo(kernel_channels_mapping, activation_min_max_mapping, {})

            If a layer's activations should be quantized and we know it's min/max outputs range in advanced, we can add it to layer_min_max_mapping for saving the statistics collection time. For example:

            >>> layer_min_max_mapping = {tf.keras.layers.Softmax: (0, 1)}
            >>> FrameworkInfo(kernel_channels_mapping, activation_min_max_mapping, layer_min_max_mapping)

        """

        self.activation_quantizer_mapping = activation_quantizer_mapping
        self.kernel_channels_mapping = kernel_channels_mapping
        self.activation_min_max_mapping = activation_min_max_mapping
        self.layer_min_max_mapping = layer_min_max_mapping
        self.kernel_ops_attributes_mapping = kernel_ops_attributes_mapping
        self.out_channel_axis_mapping = out_channel_axis_mapping

    def get_kernel_op_attributes(self, node_type: Any) -> List[str]:
        """
        Get a list of attributes of a layer's weights to quantize.

        Args:
            node_type: Layer to get its attributes.

        Returns:
            A list of attributes the layer has and should be quantized.
        """
        attr_list = self.kernel_ops_attributes_mapping.get(node_type)
        return attr_list

    def is_kernel_op(self, node_type: Any) -> bool:
        """
        Check is the node is a kernel operation.

        Args:
            node_type: Layer to get its attributes.

        Returns:
            True if node type is a kernel operation, else False.
        """
        return node_type in self.kernel_ops_attributes_mapping.keys()

    def layers_has_min_max(self, layer: Any) -> bool:
        """
        Check if a layer is in a layer to min/max mapping the FrameworkInfo holds.
        Args:
            layer: A layer to check if has a min/max known values.

        Returns:
            Whether a layer has a min/max known values or not.
        """

        return layer in self.layer_min_max_mapping

    def activation_has_min_max(self, activation_name: str) -> bool:
        """
        Check if an activation layer has a min/max mapping.

        Args:
            activation_name: String of the activation function to check for its min/max values.

        Returns:
            Whether an activation layer has a min/max known values or not.
        """

        return activation_name in self.activation_min_max_mapping
