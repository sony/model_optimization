import copy

import keras.layers
import numpy as np

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode

def ddd(node: BaseNode,
         input_mask: np.ndarray,
         output_mask: np.ndarray,
         fw_info: FrameworkInfo):
    _node = copy.deepcopy(node)


# Get the number of parameters for a pruned Keras node.
def get_keras_pruned_node_num_params(node: BaseNode,
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

                # # Special handling for Dense layers to align input mask with kernel shape.
                # if node.type == keras.layers.Dense:
                #     if w.shape[ic_axis] != len(input_mask):
                #         num_ic_per_prev_oc_channel = w.shape[ic_axis] / len(input_mask)
                #         assert int(num_ic_per_prev_oc_channel) == num_ic_per_prev_oc_channel
                #         input_mask = np.repeat(input_mask, int(num_ic_per_prev_oc_channel))

                # Assert the input and output masks match the kernel dimensions.
                assert w.shape[ic_axis] == len(input_mask), (
                    f"Kernel num of input channels: {w.shape[ic_axis]}, but mask len is {len(input_mask)} for node "
                    f"{node}")
                assert w.shape[oc_axis] == len(
                    output_mask), (f"Kernel num of output channels: {w.shape[oc_axis]}, but mask len is "
                                   f"{len(output_mask)} for node {node}")

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