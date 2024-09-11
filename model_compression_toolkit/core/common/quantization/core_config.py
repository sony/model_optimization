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
from dataclasses import dataclass, field
from typing import Optional

from model_compression_toolkit.core.common.quantization.bit_width_config import BitWidthConfig
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core.common.quantization.debug_config import DebugConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import MixedPrecisionQuantizationConfig


@dataclass
class CoreConfig:
    """
    A dataclass to hold the configurations classes of the MCT-core.

    Args:
        quantization_config (QuantizationConfig): Config for quantization.
        mixed_precision_config (MixedPrecisionQuantizationConfig): Config for mixed precision quantization.
            If None, a default MixedPrecisionQuantizationConfig is used.
        bit_width_config (BitWidthConfig): Config for manual bit-width selection.
        debug_config (DebugConfig): Config for debugging and editing the network quantization process.
    """

    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig)
    mixed_precision_config: MixedPrecisionQuantizationConfig = field(default_factory=MixedPrecisionQuantizationConfig)
    bit_width_config: BitWidthConfig = field(default_factory=BitWidthConfig)
    debug_config: DebugConfig = field(default_factory=DebugConfig)

    @property
    def is_mixed_precision_enabled(self) -> bool:
        """
        A property that indicates whether mixed precision is enabled.
        """
        return bool(self.mixed_precision_config and self.mixed_precision_config.is_mixed_precision_enabled)

