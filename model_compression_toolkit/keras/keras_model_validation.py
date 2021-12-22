from tensorflow.keras.models import Model

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.common.model_validation import ModelValidation
from model_compression_toolkit.keras.constants import CHANNELS_FORMAT, CHANNELS_FORMAT_LAST, CHANNELS_FORMAT_FIRST


class KerasModelValidation(ModelValidation):

    def __init__(self, model: Model, fw_info: FrameworkInfo):
        super(KerasModelValidation, self).__init__(model=model,
                                                   fw_info=fw_info)

    def validate_output_channel_consistency(self):
        for layer in self.model.layers:
            data_format = layer.get_config().get(CHANNELS_FORMAT)
            if data_format is not None:
                assert (data_format == CHANNELS_FORMAT_LAST and self.fw_info.output_channel_index == -1
                        or data_format == CHANNELS_FORMAT_FIRST and self.fw_info.output_channel_index == 1)
