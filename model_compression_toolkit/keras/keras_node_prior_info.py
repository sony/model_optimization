from typing import Any, Tuple

import tensorflow as tf
if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Activation, ReLU
else:
    from keras.layers import Activation, ReLU

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.common import BaseNode
from model_compression_toolkit.common.node_prior_info import NodePriorInfo
from model_compression_toolkit.keras.constants import ACTIVATION, RELU_MAX_VALUE, NEGATIVE_SLOPE, THRESHOLD


def create_node_prior_info(node: BaseNode,
                           fw_info: FrameworkInfo):
    """
    Create a NodePriorInfo object for a given node.

    Args:
        node: Node to create its prior info.
        fw_info: Information about a specific framework the node was generated from.

    Returns:
        NodePriorInfo object with info about the node.
    """

    min_output, max_output = _get_min_max_outputs(node=node,
                                                  fw_info=fw_info)
    return NodePriorInfo(min_output=min_output,
                         max_output=max_output)


def _get_min_max_outputs(node: BaseNode,
                         fw_info: FrameworkInfo) -> Tuple[Any,Any]:
    """
    Return the min/max output values of a node if known.
    If one of them (or both of them) is unknown - return None instead of a value.
    Args:
        node: Node to create its prior info.
        fw_info: Information about a specific framework the node was generated from.

    Returns:
        Min/max output values if known.
    """
    min_output, max_output = None, None

    if node.type == ReLU:
        min_output = node.framework_attr[THRESHOLD] if node.framework_attr[NEGATIVE_SLOPE] == 0 else None
        max_output = node.framework_attr[RELU_MAX_VALUE]

    elif fw_info.layers_has_min_max(node.type):
        min_output, max_output = fw_info.layer_min_max_mapping[node.type]

    elif node.type == Activation and fw_info.activation_has_min_max(node.framework_attr[ACTIVATION]):
        min_output, max_output = fw_info.activation_min_max_mapping[node.framework_attr[ACTIVATION]]

    return min_output, max_output


