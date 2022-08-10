from keras import Model
from keras.engine.input_layer import InputLayer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper


def validate_complete_quantization_info(model: Model):
    for layer in model.layers:
        assert isinstance(layer, QuantizeWrapper) or isinstance(layer, InputLayer)