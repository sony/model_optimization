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
from typing import Dict, Any, List

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.core.common.quantization.node_quantization_config import BaseNodeQuantizationConfig
from model_compression_toolkit.core.common.target_platform import QuantizationMethod

from model_compression_toolkit.quantizers_infrastructure.common.base_trainable_quantizer import BaseTrainableQuantizer
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget

if FOUND_TORCH:

    class BasePytorchTrainableQuantizer(BaseTrainableQuantizer):
        def __init__(self,
                     quantization_config: BaseNodeQuantizationConfig,
                     quantization_target: QuantizationTarget,
                     quantization_method: List[QuantizationMethod]):
            """
            This class is a base Pytorch quantizer which validates the provided quantization config and defines an
            abstract function which any quantizer needs to implement.

            Args:
                quantization_config: node quantization config class contains all the information about the quantizer.
                quantization_target: A enum which selects the quantizer tensor type: activation or weights.
                quantization_method: A list of enums which represent the supported methods for the quantizer.
            """
            super().__init__(quantization_config, quantization_target, quantization_method)

else:
    class BasePytorchTrainableQuantizer(BaseTrainableQuantizer):
        def __init__(self, quantization_config: BaseNodeQuantizationConfig, quantization_target: QuantizationTarget,
                     quantization_method: List[QuantizationMethod]):
            super().__init__(quantization_config, quantization_target, quantization_method)
            Logger.critical('Installing Pytorch is mandatory '
                            'when using BasePytorchQuantizer. '
                            'Could not find torch package.')
