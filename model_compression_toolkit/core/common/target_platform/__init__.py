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

from model_compression_toolkit.core.common.target_platform.fusing import Fusing
from model_compression_toolkit.core.common.target_platform.targetplatform2framework import \
    TargetPlatformCapabilities, OperationsSetToLayers, Smaller, SmallerEq, NotEq, Eq, GreaterEq, Greater, LayerFilterParams, OperationsToLayers, get_current_tpc

from model_compression_toolkit.core.common.target_platform.target_platform_model import \
    get_default_quantization_config_options, TargetPlatformModel

from model_compression_toolkit.core.common.target_platform.op_quantization_config import OpQuantizationConfig, \
    QuantizationConfigOptions, QuantizationMethod
from model_compression_toolkit.core.common.target_platform.operators import OperatorsSet, OperatorSetConcat



