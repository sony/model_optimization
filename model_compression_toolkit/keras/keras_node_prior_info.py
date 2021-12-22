from typing import Any, Tuple

from keras.layers import Activation, ReLU

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.common import BaseNode
from model_compression_toolkit.common.base_node_prior_info import BaseNodePriorInfo
from model_compression_toolkit.common.collectors.statistics_collector import is_number
from model_compression_toolkit.keras.constants import ACTIVATION, RELU_MAX_VALUE, NEGATIVE_SLOPE, THRESHOLD


class KerasNodePriorInfo(BaseNodePriorInfo):

    def __init__(self,
                 node: BaseNode,
                 fw_info: FrameworkInfo):
        super(KerasNodePriorInfo, self).__init__(node=node,
                                                 fw_info=fw_info)

    def is_output_bounded(self):
        min_output, max_output = self.get_min_max()
        return is_number(min_output) and is_number(max_output)

    def get_min_max(self) -> Tuple[Any,Any]:
        min_output, max_output = None, None
        if self.node.layer_class == ReLU:
            min_output = self.node.framework_attr[THRESHOLD] if self.node.framework_attr[NEGATIVE_SLOPE] == 0 else None
            max_output = self.node.framework_attr[RELU_MAX_VALUE]
        elif self.fw_info.layers_has_min_max(self.node.layer_class):
            min_output, max_output = self.fw_info.layer_min_max_mapping[self.node.layer_class]
        elif self.node.layer_class == Activation and self.fw_info.activation_has_min_max(self.node.framework_attr[ACTIVATION]):
            min_output, max_output = self.fw_info.activation_min_max_mapping[self.node.framework_attr[ACTIVATION]]

        return min_output, max_output

