# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

import keras.layers
import numpy as np

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode





def prune_keras_entry_node(node: BaseNode,
                           output_mask: np.ndarray,
                           fw_info: FrameworkInfo):
    """

    Args:
        node:
        output_mask:
        fw_info:

    Returns:

    """
    return _prune_keras_edge_node(node=node,
                                  mask=output_mask,
                                  fw_info=fw_info,
                                  is_exit_node=False)


def prune_keras_intermediate_node(node: BaseNode,
                                  input_mask: np.ndarray,
                                  output_mask: np.ndarray,
                                  fw_info: FrameworkInfo):
    """

    Args:
        node:
        input_mask:
        output_mask:
        fw_info:

    Returns:

    """
    _edit_node_input_shape(input_mask, node)
    pruned_parameters = {}
    mask_bool = output_mask.astype(bool)
    for k, v in node.weights.items():
        # Apply the mask to the weights.
        pruned_parameters[k] = v.compress(mask_bool, axis=-1)
    node.weights = pruned_parameters


def prune_keras_exit_node(node: BaseNode,
                          input_mask: np.ndarray,
                          fw_info: FrameworkInfo):
    """

    Args:
        node:
        input_mask:
        fw_info:

    Returns:

    """
    return _prune_keras_edge_node(node=node,
                                  mask=input_mask,
                                  fw_info=fw_info,
                                  is_exit_node=True)


def _edit_node_input_shape(input_mask, node):
    """

    Args:
        input_mask:
        node:

    Returns:

    """
    new_input_shape = list(node.input_shape)
    # The last dimension of the input shape is adjusted based on the sum of the mask.
    new_input_shape[-1] = int(np.sum(input_mask))
    node.input_shape = tuple(new_input_shape)


def _prune_keras_edge_node(node: BaseNode,
                           mask: np.ndarray,
                           fw_info: FrameworkInfo,
                           is_exit_node: bool):
    """
    Prunes the given Keras node by applying the mask to the node's weights (kernels and biases).
    This function can handle both entry and exit nodes by specifying the is_exit_node parameter.

    Args:
        node: The node to be pruned.
        mask: The pruning mask to be applied.
        fw_info: Framework-specific information object.
        is_exit_node: A boolean indicating whether the node is an exit node.

    Returns:
        None
    """

    # Retrieve the kernel attribute and the axes to prune.
    kernel_attr = fw_info.get_kernel_op_attributes(node.type)[0]
    io_axis = fw_info.kernel_channels_mapping.get(node.type)
    axis_to_prune = io_axis[int(is_exit_node)]
    kernel = node.get_weights_by_keys(kernel_attr)
    # Convert mask to boolean.
    mask_bool = mask.astype(bool)

    # # Special handling for Dense layers at the exit of a pruning section.
    # if is_exit_node and node.type == keras.layers.Dense:
    #     num_ic_per_prev_oc_channel = kernel.shape[axis_to_prune] / len(mask_bool)
    #     assert int(num_ic_per_prev_oc_channel) == num_ic_per_prev_oc_channel
    #     mask_bool = np.repeat(mask_bool, int(num_ic_per_prev_oc_channel))

    pruned_kernel = kernel.compress(mask_bool, axis=axis_to_prune)
    node.set_weights_by_keys(name=kernel_attr, tensor=pruned_kernel)

    if not is_exit_node and node.framework_attr['use_bias']:
        # Prune the bias if applicable and it's an entry node.
        bias = node.get_weights_by_keys('bias')
        pruned_bias = bias.compress(mask_bool)
        node.set_weights_by_keys(name='bias', tensor=pruned_bias)

    if not is_exit_node:
        # Update 'filters' or 'units' attributes for entry node Conv2D/Conv2DTranspose layers.
        if node.type in [keras.layers.Conv2D, keras.layers.Conv2DTranspose]:
            node.framework_attr['filters'] = int(np.sum(mask))
        elif node.type == keras.layers.Dense:
            node.framework_attr['units'] = int(np.sum(mask))

    if is_exit_node:
        # Adjust the input shape for the last node in the section.
        _edit_node_input_shape(mask_bool, node)

