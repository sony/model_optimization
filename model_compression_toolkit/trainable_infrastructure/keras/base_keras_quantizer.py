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
from typing import Dict, Any, Union, List

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import BaseTrainableQuantizer, VAR, GROUP
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig, \
    TrainableQuantizerActivationConfig

if FOUND_TF:
    QUANTIZATION_CONFIG = 'quantization_config'
    from model_compression_toolkit.trainable_infrastructure.keras.config_serialization import config_serialization, \
        config_deserialization
    import tensorflow as tf

    class BaseKerasTrainableQuantizer(BaseTrainableQuantizer):
        def __init__(self,
                     quantization_config: Union[TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig]):
            """
            This class is a base quantizer which validates provided quantization config and defines an abstract function which any quantizer needs to implement.
            This class adds to the base quantizer a get_config and from_config functions to enable loading and saving the keras model.

            Args:
                quantization_config: quantizer config class contains all the information about a quantizer configuration.
            """
            super().__init__(quantization_config)

        def get_config(self) -> Dict[str, Any]:
            """

            Returns: Configuration of BaseKerasQuantizer.

            """
            return {QUANTIZATION_CONFIG: config_serialization(self.quantization_config)}

        @classmethod
        def from_config(cls, config: dict):
            """

            Args:
                config(dict): dictonory  of  BaseKerasQuantizer Configuration

            Returns: A BaseKerasQuantizer

            """
            config = config.copy()
            quantization_config = config_deserialization(config[QUANTIZATION_CONFIG])
            # Note that a quantizer only receive quantization config and the rest of define hardcoded inside the speficie quantizer.
            return cls(quantization_config=quantization_config)

        def get_trainable_variables(self, group: VariableGroup) -> List[tf.Tensor]:
            """
            Get trainable parameters with specific group from quantizer

            Args:
                group: Enum of variable group

            Returns:
                List of trainable variables
            """
            quantizer_trainable = []
            for name, parameter_dict in self.quantizer_parameters.items():
                quantizer_parameter, parameter_group = parameter_dict[VAR], parameter_dict[GROUP]
                if quantizer_parameter.trainable and parameter_group == group:
                    quantizer_trainable.append(quantizer_parameter)
            return quantizer_trainable


else:
    class BaseKerasTrainableQuantizer(BaseTrainableQuantizer):
        def __init__(self, *args, **kwargs):
            Logger.critical("Tensorflow must be installed to use BaseKerasTrainableQuantizer. "
                            "The 'tensorflow' package is missing.")  # pragma: no cover
