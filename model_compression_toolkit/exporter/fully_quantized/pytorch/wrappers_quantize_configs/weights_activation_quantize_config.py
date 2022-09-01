from model_compression_toolkit.exporter.fully_quantized.pytorch.wrappers_quantize_configs.quantize_config import \
    QuantizeConfig


class WeightsActivationQuantizeConfig(QuantizeConfig):

    def __init__(self,
                 weight_quantizers,
                 activation_quantizers):
        super().__init__(is_weight_quantized=True,
                         is_activation_quantized=True)

        self._weight_quantizers = weight_quantizers
        self._activation_quantizers = activation_quantizers

    def get_weight_quantizers(self):
        return self._weight_quantizers

    def get_activation_quantizers(self):
        return self._activation_quantizers



