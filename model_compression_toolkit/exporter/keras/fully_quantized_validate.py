from keras import Model
from keras.engine.input_layer import InputLayer
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import \
    NoOpQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit.exporter.keras.quantize_configs.activation_quantize_config import \
    ActivationQuantizeConfig
from model_compression_toolkit.exporter.keras.quantize_configs.weights_activation_quantize_config import \
    WeightsActivationQuantizeConfig
from model_compression_toolkit.exporter.keras.quantize_configs.weights_quantize_config import WeightsQuantizeConfig

supported_quantize_configs = [WeightsQuantizeConfig,
                              ActivationQuantizeConfig,
                              WeightsActivationQuantizeConfig,
                              NoOpQuantizeConfig]


def fully_quantized_validate(model: Model):
    for layer in model.layers:
        assert isinstance(layer, QuantizeWrapper) or isinstance(layer, InputLayer)
        if isinstance(layer, QuantizeWrapper):
            assert type(layer.quantize_config) in supported_quantize_configs, f'Layer\'s quantize_config is not ' \
                                                                              f'supported by ' \
                                                                              f'exporter: {type(layer.quantize_config)}'
