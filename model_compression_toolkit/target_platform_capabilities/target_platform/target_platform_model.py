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

from model_compression_toolkit.target_platform_capabilities.target_platform.current_tp_model import get_current_tp_model
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import QuantizationConfigOptions


def get_default_quantization_config_options() -> QuantizationConfigOptions:
    """

    Returns: The default QuantizationConfigOptions of the model. This is the options
    to use when a layer's options is queried and it wasn't specified in the TargetPlatformCapabilities.
    The default QuantizationConfigOptions always contains a single option.

    """
    return get_current_tp_model().default_qco


