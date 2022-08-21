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


from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import \
    NoOpQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_registry import \
    QuantizeConfig

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.qat.keras.quantizer.configs.weight_quantizer_config import WeightQuantizeConfig


QUANTIZATION_CONFIGS_DICT = {"WeightQuantizeConfig": WeightQuantizeConfig}


def quantization_config_builder(n: common.BaseNode,
                                fw_info: FrameworkInfo) -> QuantizeConfig:
    """
    Build a QuantizeConfig for a node according to its quantization configuration and
    a global NoOpQuantizeConfig object.

    Args:
        n: Node to build its QuantizeConfig.
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).

    Returns:
        A QuantizeConfig object with the appropriate quantizers (according to the node's
        quantization configuration).
    """

    if n.is_weights_quantization_enabled():
        qc = WeightQuantizeConfig(fw_info.get_kernel_op_attributes(n.type),
                                  n.final_weights_quantization_cfg.weights_n_bits,
                                  n.final_weights_quantization_cfg.weights_channels_axis,
                                  n.final_weights_quantization_cfg.weights_quantization_method,
                                  n.final_weights_quantization_cfg.weights_quantization_params)
    else:
        qc = NoOpQuantizeConfig()

    return qc
