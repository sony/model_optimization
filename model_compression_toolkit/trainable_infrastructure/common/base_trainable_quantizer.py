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
from abc import abstractmethod
from enum import Enum
from typing import Union, List, Any
from inspect import signature

from model_compression_toolkit.core import common
from model_compression_toolkit.logger import Logger

from mct_quantizers.common.base_inferable_quantizer import BaseInferableQuantizer, \
    QuantizationTarget
from model_compression_toolkit.trainable_infrastructure.common.trainable_quantizer_config import \
    TrainableQuantizerActivationConfig, TrainableQuantizerWeightsConfig
from mct_quantizers.common.constants import QUANTIZATION_METHOD, \
    QUANTIZATION_TARGET


VAR = 'var'
GROUP = 'group'

class VariableGroup(Enum):
    """
    An enum for choosing trainable variable group
    0. WEIGHTS
    1. QPARAMS
    """
    WEIGHTS = 0
    QPARAMS = 1


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
                    Logger.critical(f"The first parameter must be either 'TrainableQuantizerWeightsConfig' or 'TrainableQuantizerActivationConfig'.")  # pragma: no cover
            elif v.default is v.empty:
                Logger.critical(f"Parameter '{k}' lacks a default value.")  # pragma: no cover

        super(BaseTrainableQuantizer, self).__init__()
        self.quantization_config = quantization_config

        # Inherited class should be decorated with @mark_quantizer decorator, and define the following static properties
        static_quantization_method = getattr(self, QUANTIZATION_METHOD, None)
        static_quantization_target = getattr(self, QUANTIZATION_TARGET, None)

        if static_quantization_method is None or static_quantization_target is None:
            Logger.critical("Quantizer class inheriting from 'BaseTrainableQuantizer' is improperly defined. "
                            "Ensure it includes the '@mark_quantizer' decorator and is correctly applied.")

        if static_quantization_target == QuantizationTarget.Weights:
            self.validate_weights()
            if self.quantization_config.weights_quantization_method not in static_quantization_method:
                Logger.critical(
                    f"Quantization method mismatch. Expected methods: {static_quantization_method}, received: {self.quantization_config.weights_quantization_method}.")
        elif static_quantization_target == QuantizationTarget.Activation:
            self.validate_activation()
            if self.quantization_config.activation_quantization_method not in static_quantization_method:
                Logger.critical(
                    f"Quantization method mismatch. Expected methods: {static_quantization_method}, received: {self.quantization_config.activation_quantization_method}.")
        else:
            Logger.critical(
                f"Unrecognized 'QuantizationTarget': {static_quantization_target}.")  # pragma: no cover

        self.quantizer_parameters = {}

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
            Logger.critical(f'Expected weight quantization configuration; received activation quantization instead.')

    def validate_activation(self) -> None:
        """
        This function validates the quantization config compared with its parameters.

        """
        if not self.activation_quantization() or self.weights_quantization():
            Logger.critical(f'Expected activation quantization configuration; received weight quantization instead.')

    def convert2inferable(self) -> BaseInferableQuantizer:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            BaseInferableQuantizer object.
        """
        raise NotImplemented  # pragma: no cover

    def add_quantizer_variable(self, name: str, variable: Any, group: VariableGroup = VariableGroup.WEIGHTS):
        """
        Add a quantizer variable to quantizer_parameters dictionary
        """
        self.quantizer_parameters.update({name: {VAR: variable, GROUP: group}})

    def get_quantizer_variable(self, name: str) -> Any:
        """
        Get a quantizer variable by name

        Args:
            name: variable name

        Returns:
            trainable variable
        """
        if name in self.quantizer_parameters:
            return self.quantizer_parameters[name][VAR]
        else:
            Logger.critical(f"Variable '{name}' does not exist in quantizer parameters.") # pragma: no cover


    @abstractmethod
    def get_trainable_variables(self, group: VariableGroup) -> List[Any]:
        """
        Get trainable parameters with specific group from quantizer

        Args:
            group: Enum of variable group

        Returns:
            List of trainable variables
        """
        raise NotImplemented  # pragma: no cover