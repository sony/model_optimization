from model_compression_toolkit.core.pytorch.back2framework.quantization_wrapper.wrapper_quantize_config import \
    WrapperQuantizeConfig




class ActivationQuantizeConfig(WrapperQuantizeConfig):

    def __init__(self,
                 activation_quantizers):
        super().__init__(is_weight_quantized=False,
                         is_activation_quantized=True)

        self._activation_quantizers = activation_quantizers

    def get_weight_quantizers(self):
        return []

    def get_activation_quantizers(self):
        return self._activation_quantizers



