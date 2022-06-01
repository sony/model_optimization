# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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


from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core.common.quantization.debug_config import DebugConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import MixedPrecisionQuantizationConfigV2


class CoreConfig:
    """
    A class to hold the configurations classes of the MCT-core.
    """
    def __init__(self, n_iter: int = 500,
                 quantization_config: QuantizationConfig = QuantizationConfig(),
                 mixed_precision_config: MixedPrecisionQuantizationConfigV2 = None,
                 debug_config: DebugConfig = DebugConfig()
                 ):
        """

        Args:
            n_iter (int): Number of calibration iterations to run.
            quantization_config (QuantizationConfig): Config for quantization.
            mixed_precision_config (MixedPrecisionQuantizationConfigV2): Config for mixed precision quantization (optional, default=None).
            debug_config (DebugConfig): Config for debugging and editing the network quantization process.
        """
        self.n_iter = n_iter
        self.quantization_config = quantization_config
        self.mixed_precision_config = mixed_precision_config
        self.debug_config = debug_config

    @property
    def mixed_precision_enable(self):
        return self.mixed_precision_config is not None
