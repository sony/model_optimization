# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Union, Dict, List

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.gptq.common.gptq_constants import WEIGHTS_QUANTIZATION_PARAMS

from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import BaseTrainableQuantizer

if FOUND_TF:
    import tensorflow as tf

    from model_compression_toolkit.trainable_infrastructure import BaseKerasTrainableQuantizer
    from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper

    class BaseKerasGPTQTrainableQuantizer(BaseKerasTrainableQuantizer):
        """
        A base class for trainable Keras quantizer for GPTQ.
        """

        def __init__(self,
                     quantization_config: Union[TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig]):
            """
            Initializes BaseKerasGPTQTrainableQuantizer object.

            Args:
                quantization_config: quantizer config class contains all the information about a quantizer configuration.
            """

            super().__init__(quantization_config)

        def update_layer_quantization_params(self, layer: KerasTrainableQuantizationWrapper
                                             ) -> (Dict[str, tf.Tensor], Dict[str, Dict], Dict):
            """
            A Function to calculate the needed change in attributes in NodeQuantizationConfig after retraining.

            Args:
                layer: A wrapped Keras layer.

            Returns:
                3 dictionaries describing the change in layer's weights, weights config, activation config
                that changed during GPTQ retraining.
                Keys must match NodeQuantizationConfig attributes

            """
            weights = {}
            for weight, quantizer_vars, quantizer in layer.get_weights_vars():
                if not isinstance(quantizer, BaseTrainableQuantizer):
                    Logger.critical(f"Expecting a GPTQ trainable quantizer for layer '{layer.name}', but received {type(quantizer)}. "
                                    f"Ensure a trainable quantizer is used.") # pragma: no cover
                weights.update({weight: quantizer(training=False, inputs=quantizer_vars)})

            quant_config = {WEIGHTS_QUANTIZATION_PARAMS: self.get_quant_config()}

            return weights, quant_config, {}

        def get_aux_variable(self) -> List[tf.Tensor]:
            """
            This function return a list with the quantizer's quantization auxiliary variables.

            Returns: A list with the quantization auxiliary variables.

            """

            return []  # pragma: no cover

        def get_quantization_variable(self) -> List[tf.Tensor]:
            """
            This function return a list with the quantizer's quantization parameters variables.

            Returns: A list with the quantization parameters.

            """

            return []  # pragma: no cover

        @abstractmethod
        def get_quant_config(self):
            """
            Returns the config used to edit NodeQuantizationConfig after GPTQ retraining.

            Returns:
                A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
                Keys must match NodeQuantizationConfig attributes.

            """
            raise NotImplemented(f'{self.__class__.__name__} have to implement the '  # pragma: no cover
                                 f'quantizer\'s get_quant_config.')

else:
    class BaseKerasGPTQTrainableQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            Logger.critical("Tensorflow must be installed to use BaseKerasGPTQTrainableQuantizer. "
                            "The 'tensorflow' package is missing.")  # pragma: no cover
