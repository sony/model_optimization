from model_compression_toolkit.exporter.fully_quantized.pytorch.wrappers_quantize_configs.quantize_config import \
    QuantizeConfig


class ActivationQuantizeConfig(QuantizeConfig):

    def __init__(self,
                 activation_quantizers):
        super().__init__(is_weight_quantized=False,
                         is_activation_quantized=True)

        self._activation_quantizers = activation_quantizers

    def get_weight_quantizers(self):
        return []

    def get_activation_quantizers(self):
        return self._activation_quantizers



