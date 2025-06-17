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
from typing import Dict, Any, Tuple, NamedTuple
from abc import ABC, abstractmethod

from mct_quantizers import QuantizationMethod


# Default value to use for ops without kernel.
# This is a weird default, but it's used all over the place, so for now only extract it to const so that it can be
# referenced by variable instead of hard-coded.
DEFAULT_KERNEL_ATTRIBUTE = None


class ChannelAxis(Enum):
    """

    Index of output channels axis:

    NHWC - Output channels index is last.

    NCHW - Output channels index is 1.

    """
    NHWC = -1
    NCHW = 1


class ChannelAxisMapping(NamedTuple):
    output: int
    input: int


class FrameworkInfo(ABC):
    """
    A class to wrap all information about a specific framework the library needs to quantize a model.
    Specifically, FrameworkInfo holds lists of layers by how they should be quantized, and multiple mappings such as
    layer to it kernel channels indices, and a layer to its min/max values, etc.
    The layers lists are divided into three groups:
    kernel_ops: Layers that have coefficients and need to get quantized (e.g., Conv2D, Dense, etc.)
    activation_ops: Layers that their outputs should get quantized (e.g., Add, ReLU, etc.)
    no_quantization_ops:Layers that should not get quantized (e.g., Reshape, Transpose, etc.)

    Fields:
        activation_quantizer_mapping (Dict[QuantizationMethod, Callable]): A dictionary mapping from QuantizationMethod to a quantization function.
        kernel_channels_mapping (Dict): Dictionary from a layer to a tuple of its kernel in/out channels indices.
        kernel_ops_attribute_mapping (Dict): Dictionary from a framework operator to its weight attribute to quantize.
        out_channel_axis_mapping (Dict): Dictionary of output channels of the model's layers (for computing statistics per-channel).
        _layer_min_max_mapping (Dict[Any, tuple]): Dictionary from a layer to its min/max output values.

    """

    activation_quantizer_mapping: Dict[QuantizationMethod, Callable]
    kernel_channels_mapping: Dict[Any, ChannelAxisMapping]
    kernel_ops_attribute_mapping: Dict[Any, str]
    out_channel_axis_mapping: Dict[Any, int]
    _layer_min_max_mapping: Dict[Any, tuple]

    _default_channel_mapping = ChannelAxisMapping(None, None)

    @classmethod
    def get_kernel_op_attribute(cls, node_type: Any) -> str:
        """
        Get attribute of a layer's weight to quantize.

        Args:
            node_type: Layer to get its attribute.

        Returns:
            Attribute the layer has and should be quantized.
        """
        return cls.kernel_ops_attribute_mapping.get(node_type, DEFAULT_KERNEL_ATTRIBUTE)

    @classmethod
    def is_kernel_op(cls, node_type: Any) -> bool:
        """
        Check is the node is a kernel operation.

        Args:
            node_type: Layer to get its attributes.

        Returns:
            True if node type is a kernel operation, else False.
        """
        return node_type in cls.kernel_ops_attribute_mapping

    @classmethod
    def get_layer_min_max(cls, layer: Any, fw_attrs: Dict) -> Tuple[float, float]:
        """
        Return layer min/max mapping the FrameworkInfo holds.
        Args:
            layer: A layer to check if has a min/max known values.
            fw_attrs: framework attributes from framework layer.

        Returns:
            Layer's min/max known values.
        """

        if cls.layers_has_min_max(layer):
            return cls._layer_min_max_mapping[layer]
        else:
            return None, None

    @classmethod
    def layers_has_min_max(cls, layer: Any) -> bool:
        """
        Check if a layer is in a layer to min/max mapping the FrameworkInfo holds.
        Args:
            layer: A layer to check if has a min/max known values.

        Returns:
            Whether a layer has a min/max known values or not.
        """

        return layer in cls._layer_min_max_mapping

    @classmethod
    @abstractmethod
    def get_kernel_channels(cls, node_type: Any) -> ChannelAxisMapping:
        """
        Returns node's channels mapping from kernel_channels_mapping or framework specific default value.
        Args:
            node_type: A node type

        Returns:
            Node's channels mapping.
        """
        pass

    @classmethod
    @abstractmethod
    def get_out_channel_axis(cls, node_type: Any):
        """
        Returns node's output channel mapping from out_channel_axis_mapping or framework specific default value.
        Args:
            node_type: A node type.

        Returns:
            Node's output channel axis.

        """
        pass


# Pointer to current FrameworkInfo class.
_current_framework_info: type[FrameworkInfo] = None


def get_fw_info():
    """
    A common function to get the current FrameworkInfo class. Raises an error if the pointer wasn't initialized.

    Returns: FrameworkInfo class.
    """
    assert _current_framework_info is not None, "fw_info isn't initialized."
    assert issubclass(_current_framework_info, FrameworkInfo), "fw_info isn't initialized to a FrameworkInfo class."
    return _current_framework_info


def set_fw_info(fw_info: type[FrameworkInfo]):
    """
    A common function to set the current FrameworkInfo class. Raises an error if fw_info doesn't inherit from FrameworkInfo.

    Args:
        fw_info: Framework specific object implementing the FrameworkInfo.
    """
    global _current_framework_info
    assert _current_framework_info in [None, _current_framework_info], "FrameworkInfo already initialized."
    assert issubclass(fw_info, FrameworkInfo), "fw_info must inherit from FrameworkInfo."

    _current_framework_info = fw_info
