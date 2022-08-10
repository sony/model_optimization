# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig
from typing import Tuple, List, Any, Dict
from tensorflow import Tensor
import six, abc


@six.add_metaclass(abc.ABCMeta)
class BaseQuantizeConfig(QuantizeConfig):
    """
    Base QuantizeConfig to define extra API methods needed by the GPTQ post-processing.
    """

    @abc.abstractmethod
    def get_quantization_variable(self):
        """
        A Functions that get the quantization parameters such as threshold, min, max ,etc.

        Returns: A list of trainable variable

        """

    @abc.abstractmethod
    def update_layer_quantization_params(self, layer) -> Tuple[Dict[str, Any],
                                                               Dict[str, Any],
                                                               Dict[str, Any]]:
        """
        A Function to calculate the needed change in attributes in NodeQuantizationConfig after retraining.
        Usually a function of the config quantizers.

        Args:
            layer: layer being quantized.

        Returns:
            3 dictionaries of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes:
            1. layer weights
            2. weight quantization config attributes
            3. activation quantization config attributes

        """

    @abc.abstractmethod
    def get_trainable_quantizer_parameters(self) -> List[Tensor]:
        """
        A function to get a list trainable of trainable parameters for GPTQ retraining from config quantizers

        Returns:
            A list of trainable Tensors

        """
