


class QuantizeConfig:

    def __init__(self,
                 is_weight_quantized: bool,
                 is_activation_quantized: bool
                 ):

        self.is_weight_quantized = is_weight_quantized
        self.is_activation_quantized = is_activation_quantized

    def get_weight_quantizers(self):
        raise NotImplemented


    def get_activation_quantizers(self):
        raise NotImplemented



