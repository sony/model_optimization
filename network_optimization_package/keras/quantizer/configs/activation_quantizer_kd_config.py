# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================
from typing import List, Any, Dict

from tensorflow.python.training.tracking.data_structures import ListWrapper

from network_optimization_package.common.constants import THRESHOLD
from network_optimization_package.keras.quantizer.trainable_quantizer import TrainableQuantizer
from network_optimization_package.keras.quantizer.configs.base_quantizer_kd_config import BaseQuantizeConfigKD
import numpy as np
from tensorflow.python.keras.layers import Layer
from tensorflow import Tensor
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer


class ActivationQuantizeConfigKD(BaseQuantizeConfigKD):
    """
    QuantizeConfig to quantize the activations of a layer using a TrainableQuantizer.
    """

    def __init__(self,
                 activation_quantization_params: dict,
                 signed: bool,
                 num_bits: int = 8):
        """
        Initialize a TrainableQuantizer and set as the activation quantizer.

        Args:
            activation_quantization_params: Parameters to use for quantization.
            signed: Quantization range is signed or unsigned.
            num_bits: Number of bits to use for quantization.
        """
        threshold_values = activation_quantization_params.get(THRESHOLD)
        self.activation_quantizer = TrainableQuantizer(num_bits=num_bits,
                                                       per_axis=False,
                                                       threshold_values=threshold_values,
                                                       signed=signed,
                                                       trainable=False)

    def get_weights_and_quantizers(self, layer: Layer) -> list:
        return []

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        return []

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        pass  # pragma: no cover

    def set_quantize_activations(self, layer: Layer, quantize_activations: ListWrapper):
        pass  # pragma: no cover

    def get_output_quantizers(self, layer: Layer) -> List[Quantizer]:
        """
        Get the activation quantizer.
        Args:
            layer: The layer the ActivationQuantizeConfigKD wraps.

        Returns:
            The ActivationQuantizeConfigKD activation quantizer.
        """
        return [self.activation_quantizer]

    def update_layer_quantization_params(self, layer):
        """
        A Function to calculate the needed change in attributes in NodeQuantizationConfig after retraining.
        Usually a function of the config quantizers.

        Args:
            layer: layer being quantized.

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during KD retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        quant_config = {'activation_quantization_params': self.activation_quantizer.calc_quant_config(layer)}

        return {}, quant_config

    def get_trainable_quantizer_parameters(self):
        """
        A function to get a list trainable of trainable parameters for KD retraining from config quantizers

        Returns:
            A list of trainable Tensors

        """
        return self.activation_quantizer.get_trainable_parameters()

    def get_config(self) -> Dict[str, Any]:
        """Returns the config used to serialize `QuantizeConfig`."""
        return {}
