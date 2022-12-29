from typing import Dict, List

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.qunatizers_infrastructure.common.base_quantizer import BaseQuantizer
from model_compression_toolkit.qunatizers_infrastructure.common.node_quantization_dispatcher import \
    NodeQuantizationDispatcher
from model_compression_toolkit.core.common.constants import FOUND_TORCH

if FOUND_TORCH:
    ACTIVATION_QUANTIZERS = "activation_quantizers"
    WEIGHT_QUANTIZERS = "weight_quantizer"


    class PytorchNodeQuantizationDispatcher(NodeQuantizationDispatcher):
        def __init__(self, weight_quantizers: Dict[str, BaseQuantizer] = None,
                     activation_quantizers: List[BaseQuantizer] = None):
            """
            Pytorch Node quantization dispatcher collect all the quantizer of a given layer.

            Args:
                weight_quantizers: A dictionary between weight name to it quantizer .
                activation_quantizers: A list of activation quantization one for each layer output.
            """
            super().__init__(weight_quantizers, activation_quantizers)


else:
    class PytorchNodeQuantizationDispatcher(NodeQuantizationDispatcher):
        def __init__(self, weight_quantizer: Dict[str, BaseQuantizer] = None,
                     activation_quantizers: List[BaseQuantizer] = None):
            """
            Pytorch Node quantization dispatcher collect all the quantizer of a given layer.

            Args:
                weight_quantizers: A dictionary between weight name to it quantizer .
                activation_quantizers: A list of activation quantization one for each layer output.
            """
            super().__init__(weight_quantizer, activation_quantizers)
            Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using PytorchNodeQuantizationDispatcher. '
                            'Could not find Tensorflow package.')
