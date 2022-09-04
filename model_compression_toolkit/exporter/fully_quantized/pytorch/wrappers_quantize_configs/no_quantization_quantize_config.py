from model_compression_toolkit.core.pytorch.back2framework.quantization_wrapper.wrapper_quantize_config import \
    WrapperQuantizeConfig


class NoQuantizationQuantizeConfig(WrapperQuantizeConfig):

    def __init__(self):
        super().__init__(is_weight_quantized=False,
                         is_activation_quantized=False)

    def get_weight_quantizers(self):
        return []

    def get_activation_quantizers(self):
        return []



