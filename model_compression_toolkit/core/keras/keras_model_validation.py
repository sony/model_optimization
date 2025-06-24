from tensorflow.keras.models import Model

from model_compression_toolkit.core.common.framework_info import get_fw_info
from model_compression_toolkit.core.common.framework_info import ChannelAxis
from model_compression_toolkit.core.common.model_validation import ModelValidation
from model_compression_toolkit.core.keras.constants import CHANNELS_FORMAT, CHANNELS_FORMAT_LAST, CHANNELS_FORMAT_FIRST


class KerasModelValidation(ModelValidation):
    """
    Class to define validation methods in order to validate the received Keras model to quantize.
    """

    def __init__(self, model: Model):
        """
        Initialize a KerasModelValidation object.

        Args:
            model: Keras model to check its validity.
        """

        super(KerasModelValidation, self).__init__(model=model)

    def validate_output_channel_consistency(self):
        """

        Validate that output channels index in all layers of the model are the same.
        If the model has layers with different output channels index, an exception is thrown.

        """
        fw_info = get_fw_info()
        for layer in self.model.layers:
            data_format = layer.get_config().get(CHANNELS_FORMAT)
            if data_format is not None:
                assert (data_format == CHANNELS_FORMAT_LAST and fw_info.get_out_channel_axis(layer) == ChannelAxis.NHWC.value
                        or data_format == CHANNELS_FORMAT_FIRST and fw_info.get_out_channel_axis(layer) == ChannelAxis.NCHW.value), \
                    f'Model can not have layers with different data formats.'
