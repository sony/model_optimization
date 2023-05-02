from abc import abstractmethod
from typing import Any

from model_compression_toolkit.core import FrameworkInfo


class ModelValidation:
    """
    Class to define validation methods in order to validate the received model to quantize.
    """

    def __init__(self,
                 model: Any,
                 fw_info:FrameworkInfo):
        """
        Initialize a ModelValidation object.

        Args:
            model: Model to check its validity.
            fw_info: Information about the specific framework of the model.
        """
        self.model = model
        self.fw_info = fw_info

    @abstractmethod
    def validate_output_channel_consistency(self):
        """

        Validate that output channels index in all layers of the model are the same.
        If the model has layers with different output channels index, it should throw an exception.

        """
        raise NotImplemented(
            f'Framework validation class did not implement validate_output_channel_consistency')  # pragma: no cover

    def validate(self):
        """

        Run all validation methods before the quantization process starts.

        """
        self.validate_output_channel_consistency()


