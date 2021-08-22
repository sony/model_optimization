# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
from typing import List, Any, Dict
from tensorflow import Tensor
import six, abc


@six.add_metaclass(abc.ABCMeta)
class BaseQuantizeConfigKD(QuantizeConfig):
    """
    Base QuantizeConfig to define extra API methods needed by the KD post-processing.
    """

    @abc.abstractmethod
    def update_layer_quantization_params(self, layer) -> Dict[str, Any]:
        """
        A Function to calculate the needed change in attributes in NodeQuantizationConfig after retraining.
        Usually a function of the config quantizers.

        Args:
            layer: layer being quantized.

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during KD retraining.
            Keys must match NodeQuantizationConfig attributes

        """

    @abc.abstractmethod
    def get_trainable_quantizer_parameters(self) -> List[Tensor]:
        """
        A function to get a list trainable of trainable parameters for KD retraining from config quantizers

        Returns:
            A list of trainable Tensors

        """
