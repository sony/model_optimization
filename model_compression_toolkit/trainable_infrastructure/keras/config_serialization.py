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

from typing import Any, Union
from enum import Enum

import numpy as np

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.trainable_infrastructure.common.trainable_quantizer_config import \
    TrainableQuantizerActivationConfig, TrainableQuantizerWeightsConfig
from mct_quantizers.common import constants as C


CONFIG = "config"
VALUE = "value"

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


def config_serialization(quantization_config: Union[TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig]):
    """
    This function change trainable quantizer config to a dictionary
    Args:
        quantization_config: A TrainableQuantizerWeightsConfig or TrainableQuantizerActivationConfig for serialization

    Returns: A config dictionary of quantizer config

    """
    config_data = {k: transform_enum(v) for k, v in quantization_config.__dict__.items()}
    config_data[C.IS_WEIGHTS] = isinstance(quantization_config, TrainableQuantizerWeightsConfig)
    config_data[C.IS_ACTIVATIONS] = isinstance(quantization_config, TrainableQuantizerActivationConfig)
    return config_data


def config_deserialization(in_config: dict) -> Union[TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig]:
    """
    This function change config dictionary to trainable quantizer config.
    Args:
        in_config:  A config dictionary of trainable quantizer config.

    Returns: Trainable quantizer configuration object - TrainableQuantizerWeightsConfig or TrainableQuantizerActivationConfig

    """
    in_config = copy.deepcopy(in_config)
    if in_config[C.IS_WEIGHTS]:
        weights_quantization_params = {}
        for key, value in in_config[C.WEIGHTS_QUANTIZATION_PARAMS].items():
            # In TF2.13.0, serialization of numpy array is dictionary with parameters
            weights_quantization_params.update({key: np.array(value[CONFIG][VALUE] if isinstance(value, dict) else value)})
        return TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod(in_config[C.WEIGHTS_QUANTIZATION_METHOD]),
                                               weights_n_bits=in_config[C.WEIGHTS_N_BITS],
                                               weights_quantization_params=weights_quantization_params,
                                               enable_weights_quantization=in_config[C.ENABLE_WEIGHTS_QUANTIZATION],
                                               weights_channels_axis=in_config[C.WEIGHTS_CHANNELS_AXIS],
                                               weights_per_channel_threshold=in_config[C.WEIGHTS_PER_CHANNEL_THRESHOLD],
                                               min_threshold=in_config[C.MIN_THRESHOLD])
    elif in_config[C.IS_ACTIVATIONS]:
        return TrainableQuantizerActivationConfig(activation_quantization_method=QuantizationMethod(in_config[C.ACTIVATION_QUANTIZATION_METHOD]),
                                                  activation_n_bits=in_config[C.ACTIVATION_N_BITS],
                                                  activation_quantization_params=in_config[C.ACTIVATION_QUANTIZATION_PARAMS],
                                                  enable_activation_quantization=in_config[C.ENABLE_ACTIVATION_QUANTIZATION],
                                                  min_threshold=in_config[C.MIN_THRESHOLD])
    else:
        raise NotImplemented  # pragma: no cover
