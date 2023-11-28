import keras.layers
import numpy as np

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode

def is_keras_node_entry_node(node:BaseNode):
    return _is_keras_node_pruning_section_edge(node)

def is_keras_node_exit_node(node:BaseNode):
    return _is_keras_node_pruning_section_edge(node)



# Check if a Keras node is an intermediate node in a pruning section.
def is_keras_node_intermediate_pruning_section(node: BaseNode):
    # Nodes that are not Conv2D, Conv2DTranspose, DepthwiseConv2D, or Dense are considered intermediate.
    return node.type not in [keras.layers.DepthwiseConv2D,
                             keras.layers.Conv2D,
                             keras.layers.Conv2DTranspose,
                             keras.layers.Dense]


# Check if a Keras node is an edge of a pruning section.
def _is_keras_node_pruning_section_edge(node: BaseNode):
    # Convolution nodes with group=1 or Dense layers are considered edges for pruning sections.
    if node.type in [keras.layers.Conv2D, keras.layers.Conv2DTranspose]:
        return node.framework_attr['groups'] == 1
    return node.type == keras.layers.Dense


# Get the number of parameters for a pruned Keras node.
def get_keras_pruned_node_num_params(node: BaseNode,
                                     input_mask: np.ndarray,
                                     output_mask: np.ndarray,
                                     fw_info: FrameworkInfo):
    total_params = 0
    if fw_info.is_kernel_op(node.type):
        # Obtain axes info for kernel operations.
        oc_axis, ic_axis = fw_info.kernel_channels_mapping.get(node.type)
        kernel_attr = fw_info.get_kernel_op_attributes(node.type)[0]
        for w_attr, w in node.weights.items():
            # Check if the weight attribute is the kernel.
            if kernel_attr in w_attr:
                # Handle input and output masks, ensuring they are boolean arrays.
                input_mask = np.ones(w.shape[ic_axis], dtype=bool) if input_mask is None else input_mask.astype(bool)
                output_mask = np.ones(w.shape[oc_axis], dtype=bool) if output_mask is None else output_mask.astype(bool)

                # Special handling for Dense layers to align input mask with kernel shape.
                if node.type == keras.layers.Dense:
                    if w.shape[ic_axis] != len(input_mask):
                        num_ic_per_prev_oc_channel = w.shape[ic_axis] / len(input_mask)
                        assert int(num_ic_per_prev_oc_channel) == num_ic_per_prev_oc_channel
                        input_mask = np.repeat(input_mask, int(num_ic_per_prev_oc_channel))

                # Assert the input and output masks match the kernel dimensions.
                assert w.shape[ic_axis] == len(input_mask), (f"Kernel num of input channels: {w.shape[ic_axis]}, but mask len is {len(input_mask)} for node {node}")
                assert w.shape[oc_axis] == len(output_mask), f"Kernel num of output channels: {w.shape[oc_axis]}, but mask len is {len(output_mask)} for node {node}"

                # Apply masks to the kernel and calculate the remaining parameters.
                pruned_w = np.take(w, np.where(input_mask)[0], axis=ic_axis)
                pruned_w = np.take(pruned_w, np.where(output_mask)[0], axis=oc_axis)
                total_params += len(pruned_w.flatten())
            else:
                # For non-kernel weights, apply the output mask only.
                total_params += len(np.take(w, np.where(output_mask)[0]))

    else:
        # For non-kernel operations, apply the output mask to the last axis.
        for w_attr, w in node.weights.items():
            pruned_w = np.take(w, np.where(output_mask)[0], axis=-1)
            total_params += pruned_w.size

    return total_params

# Prune a Keras node.
def prune_keras_node(node: BaseNode,
                     mask: np.ndarray,
                     fw_info: FrameworkInfo,
                     last_section_node: bool = False):
    # If the node is a kernel operation, prune it as an edge node.
    if fw_info.is_kernel_op(node.type):
        _prune_edge_node(fw_info, last_section_node, mask, node)
    else:
        # If it's an intermediate node, prune accordingly.
        _prune_intermediate_node(mask, node)

# Prune an intermediate node.
def _prune_intermediate_node(mask, node):
    # Adjust the input shape of the node according to the mask.
    _edit_node_input_shape(mask, node)
    pruned_parameters = {}
    mask_bool = mask.astype(bool)
    for k, v in node.weights.items():
        # Apply the mask to the weights.
        pruned_parameters[k] = v.compress(mask_bool, axis=-1)
    node.weights = pruned_parameters

# Edit the input shape of a node based on the pruning mask.
def _edit_node_input_shape(mask, node):
    new_input_shape = list(node.input_shape)
    # The last dimension of the input shape is adjusted based on the sum of the mask.
    new_input_shape[-1] = int(np.sum(mask))
    node.input_shape = tuple(new_input_shape)

# Prune an edge node.
def _prune_edge_node(fw_info, last_section_node, mask, node):
    # Retrieve the kernel attribute and the axes to prune.
    kernel_attr = fw_info.get_kernel_op_attributes(node.type)[0]
    io_axis = fw_info.kernel_channels_mapping.get(node.type)
    axis_to_prune = io_axis[int(last_section_node)]
    kernel = node.get_weights_by_keys(kernel_attr)
    # Convert mask to boolean.
    mask_bool = mask.astype(bool)
    # Special handling for Dense layers at the edge of a pruning section.
    if last_section_node and node.type == keras.layers.Dense:
        num_ic_per_prev_oc_channel = kernel.shape[axis_to_prune] / len(mask_bool)
        assert int(num_ic_per_prev_oc_channel) == num_ic_per_prev_oc_channel
        mask_bool = np.repeat(mask_bool, int(num_ic_per_prev_oc_channel))

    # Prune the kernel using the mask along the specified axis.
    pruned_kernel = kernel.compress(mask_bool, axis=axis_to_prune)
    node.set_weights_by_keys(name=kernel_attr, tensor=pruned_kernel)
    # Prune the bias if applicable and if it's not the last section node.
    if node.framework_attr['use_bias'] and not last_section_node:
        bias = node.get_weights_by_keys('bias')
        pruned_bias = bias.compress(mask_bool)
        node.set_weights_by_keys(name='bias', tensor=pruned_bias)

    # Edit node attributes based on the mask and whether it is the last section node.
    _edit_node_attr(node, mask_bool, last_section_node)
    if last_section_node:
        # Adjust the input shape for the last node in the section.
        _edit_node_input_shape(mask_bool, node)

# Edit the attributes of a node during pruning.
def _edit_node_attr(node, mask, last_section_node):
    # Update 'filters' or 'units' attributes based on the mask for Conv2D, Conv2DTranspose, or Dense layers.
    if node.type in [keras.layers.Conv2D, keras.layers.Conv2DTranspose]:
        if not last_section_node:
            node.framework_attr['filters'] = int(np.sum(mask))
    elif node.type == keras.layers.Dense:
        if not last_section_node:
            node.framework_attr['units'] = int(np.sum(mask))
