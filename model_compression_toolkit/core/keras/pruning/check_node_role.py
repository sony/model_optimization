import keras.layers
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO


def is_keras_entry_node(node: BaseNode):
    """

    Args:
        node:

    Returns:

    """
    return _is_keras_node_pruning_section_edge(node)


def is_keras_exit_node(node: BaseNode, dual_entry_node: BaseNode):
    """

    Args:
        node:

    Returns:

    """
    return _is_keras_node_pruning_section_edge(node) and _is_same_channels(node, dual_entry_node)


def _is_same_channels(exit_node: BaseNode,
                      dual_entry_node: BaseNode):
    _, exit_ic_axis = DEFAULT_KERAS_INFO.kernel_channels_mapping.get(exit_node.type)
    entry_oc_axis, _ = DEFAULT_KERAS_INFO.kernel_channels_mapping.get(dual_entry_node.type)
    return exit_node.get_weights_by_keys('kernel').shape[exit_ic_axis] == \
        dual_entry_node.get_weights_by_keys('kernel').shape[entry_oc_axis]



# Check if a Keras node is an intermediate node in a pruning section.
def is_keras_node_intermediate_pruning_section(node: BaseNode):
    """

    Args:
        node:

    Returns:

    """
    # Nodes that are not Conv2D, Conv2DTranspose, DepthwiseConv2D, or Dense are considered intermediate.
    return node.type not in [keras.layers.DepthwiseConv2D,
                             keras.layers.Conv2D,
                             keras.layers.Conv2DTranspose,
                             keras.layers.Dense]


# Check if a Keras node is an edge of a pruning section.
def _is_keras_node_pruning_section_edge(node: BaseNode):
    """

    Args:
        node:

    Returns:

    """
    # Convolution nodes with group=1 or Dense layers are considered edges for pruning sections.
    if node.type in [keras.layers.Conv2D, keras.layers.Conv2DTranspose]:
        return node.framework_attr['groups'] == 1
    return node.type == keras.layers.Dense