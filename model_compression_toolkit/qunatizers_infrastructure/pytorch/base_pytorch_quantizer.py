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

from model_compression_toolkit.qunatizers_infrastructure.common.base_quantizer import BaseQuantizer, QuantizationTarget

if FOUND_TORCH:

    class BasePytorchQuantizer(BaseQuantizer):
        def __init__(self,
                     quantization_config: BaseNodeQuantizationConfig,
                     quantization_target: QuantizationTarget,
                     quantization_method: List[QuantizationMethod]):
            """
            This class is a base quantizer which validate the provide quantization config and define abstract function which any quantizer need to implment.

            Args:
                quantization_config: node quantization config class contins all the information above a quantizer.
                quantization_target: A enum which decided the qunaizer tensor type activation or weights.
                quantization_method: A list of enums which represent the quantizer supported methods.
            """
            super().__init__(quantization_config, quantization_target, quantization_method)


else:
    class BasePytorchQuantizer(BaseQuantizer):
        def __init__(self, quantization_config: BaseNodeQuantizationConfig, quantization_target: QuantizationTarget,
                     quantization_method: List[QuantizationMethod]):
            super().__init__(quantization_config, quantization_target, quantization_method)
            Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using BasePytorchQuantizer. '
                            'Could not find Tensorflow package.')
