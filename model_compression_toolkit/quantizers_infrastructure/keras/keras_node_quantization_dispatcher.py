from typing import Dict, List

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.quantizers_infrastructure import BaseInferableQuantizer
from model_compression_toolkit.quantizers_infrastructure.common.base_trainable_quantizer import BaseTrainableQuantizer
from model_compression_toolkit.quantizers_infrastructure.common.node_quantization_dispatcher import \
    NodeQuantizationDispatcher

if FOUND_TF:
    ACTIVATION_QUANTIZERS = "activation_quantizers"
    WEIGHT_QUANTIZERS = "weight_quantizer"
    import tensorflow as tf

    keras = tf.keras


    class KerasNodeQuantizationDispatcher(NodeQuantizationDispatcher):
        def __init__(self,
                     weight_quantizers: Dict[str, BaseInferableQuantizer] = None,
                     activation_quantizers: List[BaseInferableQuantizer] = None):
            """
            Keras Node quantization dispatcher collect all the quantizer of a given layer.
            Add to functions get_config and from_config to enable saving and loading of keras models.
            Args:
                weight_quantizers: A dictionary between a weight's name to its quantizer.
                activation_quantizers: A list of activations quantization, one for each layer output.
            """
            super().__init__(weight_quantizers,
                             activation_quantizers)

        def get_config(self) -> dict:
            """

            Returns: Configuration of KerasNodeQuantizationDispatcher.

            """
            return {
                ACTIVATION_QUANTIZERS: [keras.utils.serialize_keras_object(act) for act in self.activation_quantizers],
                WEIGHT_QUANTIZERS: {k: keras.utils.serialize_keras_object(v) for k, v in
                                    self.weight_quantizers.items()}}

        @classmethod
        def from_config(cls, config):
            """

            Args:
                config(dict): dictionary  of  KerasNodeQuantizationDispatcher Configuration

            Returns: A KerasNodeQuantizationDispatcher

            """
            config = config.copy()
            activation_quantizers = [keras.utils.deserialize_keras_object(act,
                                                                          module_objects=globals(),
                                                                          custom_objects=None) for act in
                                     config.get(ACTIVATION_QUANTIZERS)]
            weight_quantizer = {k: keras.utils.deserialize_keras_object(v,
                                                                        module_objects=globals(),
                                                                        custom_objects=None) for k, v in
                                config.get(WEIGHT_QUANTIZERS).items()}
            return cls(weight_quantizer, activation_quantizers)
else:
    class KerasNodeQuantizationDispatcher(NodeQuantizationDispatcher):
        def __init__(self,
                     weight_quantizers: Dict[str, BaseInferableQuantizer] = None,
                     activation_quantizers: List[BaseInferableQuantizer] = None):
            """
            Keras Node quantization dispatcher collect all the quantizer of a given layer.
            Add to functions get_config and from_config to enable saving and loading of keras models.
            Args:
                weight_quantizers: A dictionary between a weight's name to its quantizer.
                activation_quantizers: A list of activations quantization, one for each layer output.
            """
            super().__init__(weight_quantizers, activation_quantizers)
            Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using KerasNodeQuantizationDispatcher. '
                            'Could not find Tensorflow package.')
