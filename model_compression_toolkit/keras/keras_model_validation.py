from tensorflow.keras.models import Model

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.common.framework_info import ChannelAxis
from model_compression_toolkit.common.model_validation import ModelValidation
from model_compression_toolkit.keras.constants import CHANNELS_FORMAT, CHANNELS_FORMAT_LAST, CHANNELS_FORMAT_FIRST


class KerasModelValidation(ModelValidation):
    """
    Class to define validation methods in order to validate the received Keras model to quantize.
    """

    def __init__(self, model: Model, fw_info: FrameworkInfo):
        """
        Initialize a KerasModelValidation object.

        Args:
            model: Keras model to check its validity.
            fw_info: Information about the framework of the model (Keras).
        """

        super(KerasModelValidation, self).__init__(model=model,
                                                   fw_info=fw_info)

    def validate_output_channel_consistency(self):
        """

        Validate that output channels index in all layers of the model are the same.
        If the model has layers with different output channels index, an exception is thrown.

        """
        for layer in self.model.layers:
            data_format = layer.get_config().get(CHANNELS_FORMAT)
            if data_format is not None:
                assert (data_format == CHANNELS_FORMAT_LAST and self.fw_info.output_channel_index == ChannelAxis.NHWC
                        or data_format == CHANNELS_FORMAT_FIRST and self.fw_info.output_channel_index == ChannelAxis.NCHW), \
                    f'Model can not have layers with different data formats.'
