from typing import Dict, List

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.qunatizers_infrastructure.common.base_quantizer import BaseQuantizer
from model_compression_toolkit.qunatizers_infrastructure.common.node_quantization_dispatcher import \
    NodeQuantizationDispatcher

if FOUND_TF:
    ACTIVATION_QUANTIZERS = "activation_quantizers"
    WEIGHT_QUANTIZERS = "weight_quantizer"

    from keras.utils import deserialize_keras_object, serialize_keras_object


    class KerasNodeQuantizationDispatcher(NodeQuantizationDispatcher):
        def __init__(self, weight_quantizers: Dict[str, BaseQuantizer] = None,
                     activation_quantizers: List[BaseQuantizer] = None):
            """
            Keras Node quantization dispatcher collect all the quantizer of a given layer.
            Add to functions get_config and from_config to enable saving and loading of keras models.
            Args:
                weight_quantizers: A dictionary between weight name to it quantizer .
                activation_quantizers: A list of activation quantization one for each layer output.
            """
            super().__init__(weight_quantizers, activation_quantizers)

        def get_config(self) -> dict:
            """

            Returns: Configuration of KerasNodeQuantizationDispatcher.

            """
            return {ACTIVATION_QUANTIZERS: [serialize_keras_object(act) for act in self.activation_quantizers],
                    WEIGHT_QUANTIZERS: {k: serialize_keras_object(v) for k, v in self.weight_quantizers.items()}}

        @classmethod
        def from_config(cls, config):
            """

            Args:
                config(dict): dictonory  of  KerasNodeQuantizationDispatcher Configuration

            Returns: A KerasNodeQuantizationDispatcher

            """
            config = config.copy()
            activation_quantizers = [deserialize_keras_object(act,
                                                              module_objects=globals(),
                                                              custom_objects=None) for act in
                                     config.get(ACTIVATION_QUANTIZERS)]
            weight_quantizer = {k: deserialize_keras_object(v,
                                                            module_objects=globals(),
                                                            custom_objects=None) for k, v in
                                config.get(WEIGHT_QUANTIZERS).items()}
            return cls(weight_quantizer, activation_quantizers)
else:
    class KerasNodeQuantizationDispatcher(NodeQuantizationDispatcher):
        def __init__(self, weight_quantizer: Dict[str, BaseQuantizer] = None,
                     activation_quantizers: List[BaseQuantizer] = None):
            """
            Keras Node quantization dispatcher collect all the quantizer of a given layer.
            Add to functions get_config and from_config to enable saving and loading of keras models.
            Args:
                weight_quantizers: A dictionary between weight name to it quantizer .
                activation_quantizers: A list of activation quantization one for each layer output.
            """
            super().__init__(weight_quantizer, activation_quantizers)
            Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using KerasNodeQuantizationDispatcher. '
                            'Could not find Tensorflow package.')
