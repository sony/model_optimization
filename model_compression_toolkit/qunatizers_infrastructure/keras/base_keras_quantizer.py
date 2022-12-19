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

from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.core.common.quantization.node_quantization_config import BaseNodeNodeQuantizationConfig
from model_compression_toolkit.core.common.target_platform import QuantizationMethod

from model_compression_toolkit.qunatizers_infrastructure.common.base_quantizer import BaseQuantizer, QuantizationPart

if FOUND_TF:
    from model_compression_toolkit.qunatizers_infrastructure.keras.config_serialization import config_serialization, \
        config_deserialization


    class BaseKerasQuantizer(BaseQuantizer):
        def __init__(self, qunatization_config: BaseNodeNodeQuantizationConfig, quantization_part: QuantizationPart,
                     quantization_method: List[QuantizationMethod]):
            """

            Args:
                qunatization_config:
                quantization_part:
                quantization_method:
            """
            super().__init__(qunatization_config, quantization_part, quantization_method)

        def get_config(self) -> Dict[str, Any]:
            """

            Returns: Configuration of BaseKerasQuantizer.

            """

            return {'qunatization_config': config_serialization(self.qunatization_config)}

        @classmethod
        def from_config(cls, config: dict):
            """

            Args:
                config(dict): dictonory  of  BaseKerasQuantizer Configuration

            Returns: A BaseKerasQuantizer

            """
            config = config.copy()
            qunatization_config = config_deserialization(config['qunatization_config'])
            # Note that a quantizer only receive quantization config and the rest of define hardcoded inside the speficie quantizer.
            return cls(qunatization_config=qunatization_config)

else:
    class BaseKerasQuantizer(BaseQuantizer):
        def __init__(self, qunatization_config: BaseNodeNodeQuantizationConfig, quantization_part: QuantizationPart,
                     quantization_method: List[QuantizationMethod]):
            super().__init__(qunatization_config, quantization_part, quantization_method)
