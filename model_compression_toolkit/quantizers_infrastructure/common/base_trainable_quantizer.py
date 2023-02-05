# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Union
from inspect import signature

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Logger

from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import BaseInferableQuantizer, \
    QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.common.trainable_quantizer_config import \
    TrainableQuantizerActivationConfig, TrainableQuantizerWeightsConfig
from model_compression_toolkit.quantizers_infrastructure.common.constants import QUANTIZATION_METHOD, \
    QUANTIZATION_TARGET


class BaseTrainableQuantizer(BaseInferableQuantizer):
    def __init__(self,
                 quantization_config: Union[TrainableQuantizerActivationConfig, TrainableQuantizerWeightsConfig]):
        """
        This class is a base quantizer which validates the provided quantization config and defines an abstract function which any quantizer needs to implment.

        Args:
            quantization_config: quantizer config class contains all the information about the quantizer configuration.
        """

        # verify the quantizer class that inherits this class only has a config argument and key-word arguments
        for i, (k, v) in enumerate(self.get_sig().parameters.items()):
            if i == 0:
                if v.annotation not in [TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig]:
                    common.Logger.error(f"First parameter must be either TrainableQuantizerWeightsConfig or TrainableQuantizerActivationConfig")  # pragma: no cover
            elif v.default is v.empty:
                common.Logger.error(f"Parameter {k} doesn't have a default value")  # pragma: no cover

        super(BaseTrainableQuantizer, self).__init__()
        self.quantization_config = quantization_config

        # Inherited class should be decorated with @mark_quantizer decorator, and define the following static properties
        static_quantization_method = getattr(self, QUANTIZATION_METHOD, None)
        static_quantization_target = getattr(self, QUANTIZATION_TARGET, None)

        if static_quantization_method is None or static_quantization_target is None:
            Logger.error("A quantizer class that inherit from BaseTrainableQuantizer is not defined appropriately."
                         "Either it misses the @mark_quantizer decorator or the decorator is not used correctly.")

        if static_quantization_target == QuantizationTarget.Weights:
            self.validate_weights()
            if self.quantization_config.weights_quantization_method not in static_quantization_method:
                common.Logger.error(
                    f'Quantization method mismatch expected: {static_quantization_method} and got  {self.quantization_config.weights_quantization_method}')
        elif static_quantization_target == QuantizationTarget.Activation:
            self.validate_activation()
            if self.quantization_config.activation_quantization_method not in static_quantization_method:
                common.Logger.error(
                    f'Quantization method mismatch expected: {static_quantization_method} and got  {self.quantization_config.activation_quantization_method}')
        else:
            common.Logger.error(
                f'Unknown Quantization Part:{static_quantization_target}')  # pragma: no cover

    @classmethod
    def get_sig(cls):
        return signature(cls)

    def initialize_quantization(self,
                                tensor_shape,
                                name: str,
                                layer):
        """
        This initializes the quantizer parameters given the parameter name and shape.

        Args:
            tensor_shape:  tensor shape
            name: tensor name
            layer: layer to quantized

        Returns: None

        """
        raise NotImplemented  # pragma: no cover

    def __call__(self,
                 input2quantize,
                 training: bool):
        """
        Quantize a tensor.

        Args:
            input2quantize: Input tensor to quantize.
            training: Whether the graph is in training mode.

        Returns:
            The quantized tensor.
        """
        raise NotImplemented  # pragma: no cover

    def activation_quantization(self) -> bool:
        """

        Returns: A boolean stating is this activation quantizer

        """
        return isinstance(self.quantization_config, TrainableQuantizerActivationConfig)

    def weights_quantization(self) -> bool:
        """

        Returns: A boolean stating is this weights quantizer

        """
        return isinstance(self.quantization_config, TrainableQuantizerWeightsConfig)

    def validate_weights(self) -> None:
        """
        This function validates the quantization config compared with its parameters.


        """
        if self.activation_quantization() or not self.weights_quantization():
            common.Logger.error(f'Expect weight quantization got activation')

    def validate_activation(self) -> None:
        """
        This function validates the quantization config compared with its parameters.

        """
        if not self.activation_quantization() or self.weights_quantization():
            common.Logger.error(f'Expect activation quantization got weight')

    def convert2inferable(self) -> BaseInferableQuantizer:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            BaseInferableQuantizer object.
        """
        raise NotImplemented  # pragma: no cover
