from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common import BaseNode


class BaseNodePriorInfo:

    def __init__(self,
                 node: BaseNode,
                 fw_info: FrameworkInfo):

        self.node = node
        self.fw_info = fw_info

    def is_output_bounded(self):
        raise Exception('Framework specific prior info have to implement is_output_bounded')



