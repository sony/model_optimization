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
from model_compression_toolkit.core.common.quantization.bit_width_config import BitWidthConfig
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core.common.quantization.debug_config import DebugConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import MixedPrecisionQuantizationConfig


class CoreConfig:
    """
    A class to hold the configurations classes of the MCT-core.
    """
    def __init__(self,
                 quantization_config: QuantizationConfig = None,
                 mixed_precision_config: MixedPrecisionQuantizationConfig = None,
                 bit_width_config: BitWidthConfig = None,
                 debug_config: DebugConfig = None
                 ):
        """

        Args:
            quantization_config (QuantizationConfig): Config for quantization.
            mixed_precision_config (MixedPrecisionQuantizationConfig): Config for mixed precision quantization.
            If None, a default MixedPrecisionQuantizationConfig is used.
            bit_width_config (BitWidthConfig): Config for manual bit-width selection.
            debug_config (DebugConfig): Config for debugging and editing the network quantization process.
        """
        self.quantization_config = QuantizationConfig() if quantization_config is None else quantization_config
        self.bit_width_config = BitWidthConfig() if bit_width_config is None else bit_width_config
        self.debug_config = DebugConfig() if debug_config is None else debug_config

        if mixed_precision_config is None:
            self.mixed_precision_config = MixedPrecisionQuantizationConfig()
        else:
            self.mixed_precision_config = mixed_precision_config

    @property
    def mixed_precision_enable(self):
        return self.mixed_precision_config is not None and self.mixed_precision_config.mixed_precision_enable

