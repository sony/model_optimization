from abc import abstractmethod
from typing import Any

from model_compression_toolkit import FrameworkInfo


class ModelValidation:

    def __init__(self, model: Any, fw_info:FrameworkInfo):
        self.model = model
        self.fw_info = fw_info

    @abstractmethod
    def validate_output_channel_consistency(self):
        raise NotImplemented(f'Framework validation class did not implement validate_output_channel_consistency')

    def validate(self):
        self.validate_output_channel_consistency()


