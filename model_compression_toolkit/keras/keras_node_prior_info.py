from typing import Any, Tuple
import numpy as np
import tensorflow as tf

if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Activation, ReLU, BatchNormalization
else:
    from keras.layers import Activation, ReLU, BatchNormalization

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.common import BaseNode
from model_compression_toolkit.common.node_prior_info import NodePriorInfo
from model_compression_toolkit.keras.constants import ACTIVATION, RELU_MAX_VALUE, NEGATIVE_SLOPE, THRESHOLD, \
    GAMMA, BETA, MOVING_MEAN, MOVING_VARIANCE
from model_compression_toolkit.common.graph.base_graph import Graph


def create_node_prior_info(node: BaseNode,
                           fw_info: FrameworkInfo,
                           graph: Graph):
    """
    Create a NodePriorInfo object for a given node.

    Args:
        node: Node to create its prior info.
        fw_info: Information about a specific framework the node was generated from.
        graph: Graph to check the next node type.

    Returns:
        NodePriorInfo object with info about the node.
    """

    min_output, max_output = _get_min_max_outputs(node=node,
                                                  fw_info=fw_info)

    mean_output, std_output = _get_mean_std_outputs(node=node,
                                                    graph=graph)
    return NodePriorInfo(min_output=min_output,
                         max_output=max_output,
                         mean_output=mean_output,
                         std_output=std_output)


def _get_min_max_outputs(node: BaseNode,
                         fw_info: FrameworkInfo) -> Tuple[Any, Any]:
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


def _get_mean_std_outputs(node: BaseNode,
                          graph: Graph) -> Tuple[Any, Any]:
    """
    Return the mean/std output values of a node if known.
    If one of them (or both of them) is unknown - return None instead of a value.
    Args:
        node: Node to create its prior info.
        graph: Graph to check the next node type.

    Returns:
        Mean/Std output values if known.
    """
    mean_output, std_output = None, None

    if node.type == BatchNormalization:
        mean_output = node.get_weights_by_keys(BETA)
        if node.get_weights_by_keys(GAMMA) is None:
            std_output = 1.0
        else:
            std_output = np.abs(node.get_weights_by_keys(GAMMA))
        if mean_output is None:
            mean_output = 0.0
    else:
        next_node_list = graph.get_next_nodes(node)
        bn_nodes = [bn_node for bn_node in next_node_list if bn_node.type == BatchNormalization]
        if len(bn_nodes) != 0:
            bn_node = bn_nodes[0]
            moving_variance = bn_node.get_weights_by_keys(MOVING_VARIANCE)
            std_output = np.sqrt(moving_variance)
            mean_output = bn_node.get_weights_by_keys(MOVING_MEAN)
    return mean_output, std_output
