from model_compression_toolkit.exporter.fully_quantized.pytorch.wrappers_quantize_configs.quantize_config import \
    QuantizeConfig


class NoQuantizationQuantizeConfig(QuantizeConfig):

    def __init__(self):
        super().__init__(is_weight_quantized=False,
                         is_activation_quantized=False)

    def get_weight_quantizers(self):
        return []

    def get_activation_quantizers(self):
        return []



