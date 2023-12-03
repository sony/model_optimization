
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.keras.constants import BIAS


def get_keras_node_attributes_with_io_axis(node: BaseNode,
                                           fw_info: FrameworkInfo):
    """

    Args:
        node:
        fw_info:

    Returns:

    """
    attributes_with_axis = {}
    if fw_info.is_kernel_op(node.type):
        kernel_attributes = fw_info.get_kernel_op_attributes(node.type)
        for attr in kernel_attributes:
            attributes_with_axis[attr] = fw_info.kernel_channels_mapping.get(node.type)
        attributes_with_axis[BIAS] = (0, None)
    else:
        for attr in list(node.weights.keys()):
            attributes_with_axis[attr] = (-1, None)

    return attributes_with_axis
