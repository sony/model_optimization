from typing import Dict, List

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.quantizers_infrastructure.common.node_quantization_dispatcher import \
    NodeQuantizationDispatcher
from model_compression_toolkit.core.common.constants import FOUND_TORCH

if FOUND_TORCH:
    ACTIVATION_QUANTIZERS = "activation_quantizers"
    WEIGHT_QUANTIZERS = "weight_quantizer"

    from model_compression_toolkit.quantizers_infrastructure.pytorch.inferable_quantizers import \
        BasePyTorchInferableQuantizer

    class PytorchNodeQuantizationDispatcher(NodeQuantizationDispatcher):
        def __init__(self,
                     weight_quantizers: Dict[str, BasePyTorchInferableQuantizer] = None,
                     activation_quantizers: List[BasePyTorchInferableQuantizer] = None):
            """
            Pytorch Node quantization dispatcher collect all the quantizer of a given layer.

            Args:
                weight_quantizers: A dictionary between weight name to it quantizer .
                activation_quantizers: A list of activation quantization one for each layer output.
            """
            super().__init__(weight_quantizers, activation_quantizers)


else:
    class PytorchNodeQuantizationDispatcher(NodeQuantizationDispatcher):
        def __init__(self, *args, **kwargs):
            """
            Pytorch Node quantization dispatcher collect all the quantizer of a given layer.
            """
            Logger.critical('Installing torch is mandatory '
                            'when using PytorchNodeQuantizationDispatcher. '
                            'Could not find torch package.')
