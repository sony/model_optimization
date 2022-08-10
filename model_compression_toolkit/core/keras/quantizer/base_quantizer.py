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

from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer
from typing import List, Any, Dict
from tensorflow import Tensor
import six, abc


@six.add_metaclass(abc.ABCMeta)
class BaseTrainableQuantizer(Quantizer):
    """
    Base trainable quantizer to define extra methods needed by the GPTQ post-processing.
    """

    @abc.abstractmethod
    def get_quant_config(self, layer) -> Dict[str, Any]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Args:
            layer: quantized layer

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """

    @abc.abstractmethod
    def get_trainable_parameters(self) -> List[Tensor]:
        """
        A function to get a list trainable of trainable parameters for GPTQ retraining

        Returns:
            A list of trainable Tensors

        """

