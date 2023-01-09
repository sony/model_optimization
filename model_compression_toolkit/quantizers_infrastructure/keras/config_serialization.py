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
import copy

from collections.abc import Callable
from typing import Any

from model_compression_toolkit import QuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    BaseNodeQuantizationConfig, NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.target_platform import QuantizationMethod, OpQuantizationConfig
from enum import Enum

IS_WEIGHTS = "is_weights"
IS_ACTIVATIONS = "is_activations"
WEIGHTS_QUANTIZATION_METHOD = "weights_quantization_method"
ACTIVATIONS_QUANTIZATION_METHOD = "activation_quantization_method"


def transform_enum(v: Any):
    """
    If an enum is received it value is return otherwise the input is returned.
    Args:
        v: Any type

    Returns: Any

    """
    if isinstance(v, Enum):
        return v.value
    return v


def config_serialization(quantization_config: BaseNodeQuantizationConfig):
    """
    This function change BaseNodeQuantizationConfig to a dictionary
    Args:
        quantization_config: A BaseNodeQuantizationConfig for serialization

    Returns: A config dictionary of BaseNodeQuantizationConfig

    """
    config_data = {k: transform_enum(v) for k, v in quantization_config.__dict__.items() if
                   v is not isinstance(v, Callable)}
    config_data[IS_WEIGHTS] = isinstance(quantization_config, NodeWeightsQuantizationConfig)
    config_data[IS_ACTIVATIONS] = isinstance(quantization_config, NodeActivationQuantizationConfig)
    return config_data


def config_deserialization(in_config: dict) -> BaseNodeQuantizationConfig:
    """
    This function change config dictionary to it BaseNodeQuantizationConfig.
    Args:
        in_config:  A config dictionary of BaseNodeQuantizationConfig

    Returns: A BaseNodeQuantizationConfig

    """
    in_config = copy.deepcopy(in_config)
    qc = QuantizationConfig()
    op_cfg = OpQuantizationConfig(QuantizationMethod.POWER_OF_TWO, QuantizationMethod.POWER_OF_TWO,
                                  8, 8, True, True, True, True, 0, 0, 8)
    if in_config[IS_WEIGHTS]:
        nwqc = NodeWeightsQuantizationConfig(qc=qc,
                                             op_cfg=op_cfg,
                                             weights_quantization_fn=None,
                                             weights_quantization_params_fn=None,
                                             weights_channels_axis=0)
        in_config[WEIGHTS_QUANTIZATION_METHOD] = QuantizationMethod(in_config[WEIGHTS_QUANTIZATION_METHOD])

        nwqc.__dict__.update(in_config)
        return nwqc
    elif in_config[IS_ACTIVATIONS]:
        naqc = NodeActivationQuantizationConfig(qc=qc,
                                                op_cfg=op_cfg,
                                                activation_quantization_fn=None,
                                                activation_quantization_params_fn=None)
        in_config[ACTIVATIONS_QUANTIZATION_METHOD] = QuantizationMethod(in_config[ACTIVATIONS_QUANTIZATION_METHOD])

        naqc.__dict__.update(in_config)
        return naqc
    else:
        raise NotImplemented
