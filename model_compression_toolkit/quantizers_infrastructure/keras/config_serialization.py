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

from typing import Any
from enum import Enum

from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.common.base_trainable_quantizer_config import BaseQuantizerConfig, \
    TrainableQuantizerActivationConfig, TrainableQuantizerWeightsConfig

IS_WEIGHTS = "is_weights"
IS_ACTIVATIONS = "is_activations"


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

def config_serialization(quantization_config: BaseQuantizerConfig):
    """
    This function change BaseQuantizerConfig to a dictionary
    Args:
        quantization_config: A BaseQuantizerConfig for serialization

    Returns: A config dictionary of BaseQuantizerConfig

    """
    config_data = {k: transform_enum(v) for k, v in quantization_config.__dict__.items()}
    config_data[IS_WEIGHTS] = isinstance(quantization_config, TrainableQuantizerWeightsConfig)
    config_data[IS_ACTIVATIONS] = isinstance(quantization_config, TrainableQuantizerActivationConfig)
    return config_data


def config_deserialization(in_config: dict) -> BaseQuantizerConfig:
    """
    This function change config dictionary to it BaseQuantizerConfig.
    Args:
        in_config:  A config dictionary of BaseQuantizerConfig

    Returns: A BaseQuantizerConfig

    """
    in_config = copy.deepcopy(in_config)
    if in_config[IS_WEIGHTS]:
        return TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod(in_config['weights_quantization_method']),
                                               weights_n_bits=in_config['weights_n_bits'],
                                               weights_quantization_params=in_config['weights_quantization_params'],
                                               enable_weights_quantization=in_config['enable_weights_quantization'],
                                               weights_channels_axis=in_config['weights_channels_axis'],
                                               weights_per_channel_threshold=in_config['weights_per_channel_threshold'],
                                               min_threshold=in_config['min_threshold'])
    elif in_config[IS_ACTIVATIONS]:
        return TrainableQuantizerActivationConfig(activation_quantization_method=QuantizationMethod(in_config['activation_quantization_method']),
                                                  activation_n_bits=in_config['activation_n_bits'],
                                                  activation_quantization_params=in_config['activation_quantization_params'],
                                                  enable_activation_quantization=in_config['enable_activation_quantization'],
                                                  min_threshold=in_config['min_threshold'])
    else:
        raise NotImplemented
