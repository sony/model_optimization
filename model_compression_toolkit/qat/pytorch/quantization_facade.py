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

from typing import Callable

from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import PYTORCH
from model_compression_toolkit.core.common.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit import CoreConfig

if FOUND_TORCH:
    from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
    from model_compression_toolkit.core.pytorch.constants import DEFAULT_TP_MODEL
    from torch.nn import Module

    from model_compression_toolkit import get_target_platform_capabilities
    DEFAULT_PYTORCH_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)

    def pytorch_quantization_aware_training_init(in_module: Module,
                                                 representative_data_gen: Callable,
                                                 target_kpi: KPI = None,
                                                 core_config: CoreConfig = CoreConfig(),
                                                 fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                                                 target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_PYTORCH_TPC):
        Logger.error("Quantization Aware Training isn't supported yet.")

else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def pytorch_quantization_aware_training_init(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using pytorch_quantization_aware_training_init. '
                        'Could not find the torch package.')
