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
from model_compression_toolkit.constants import ACTIVATION_QUANTIZATION_CFG, WEIGHTS_QUANTIZATION_CFG, QC, \
    OP_CFG, ACTIVATION_QUANTIZATION_FN, WEIGHTS_QUANTIZATION_FN, ACTIVATION_QUANT_PARAMS_FN, WEIGHTS_QUANT_PARAMS_FN, \
    WEIGHTS_CHANNELS_AXIS
from model_compression_toolkit.core.common.quantization.node_quantization_config import BaseNodeQuantizationConfig, \
    NodeWeightsQuantizationConfig, NodeActivationQuantizationConfig


##########################################
# Every node holds a quantization configuration
# for its weights quantization, and a different quantization
# configuration for its activation quantization configuration.
##########################################

class CandidateNodeQuantizationConfig(BaseNodeQuantizationConfig):
    """
    Class for representing candidate node configuration, which includes weights and activation configuration combined.
    """

    def __init__(self, **kwargs):
        activation_quantization_cfg = kwargs.get(ACTIVATION_QUANTIZATION_CFG, None)
        if activation_quantization_cfg is not None:
            self.activation_quantization_cfg = activation_quantization_cfg
        else:
            self.activation_quantization_cfg = NodeActivationQuantizationConfig(kwargs.get(QC),
                                                                                kwargs.get(OP_CFG),
                                                                                kwargs.get(ACTIVATION_QUANTIZATION_FN),
                                                                                kwargs.get(ACTIVATION_QUANT_PARAMS_FN))
        weights_quantization_cfg = kwargs.get(WEIGHTS_QUANTIZATION_CFG, None)
        if weights_quantization_cfg is not None:
            self.weights_quantization_cfg = weights_quantization_cfg
        else:
            self.weights_quantization_cfg = NodeWeightsQuantizationConfig(kwargs.get(QC),
                                                                          kwargs.get(OP_CFG),
                                                                          kwargs.get(WEIGHTS_QUANTIZATION_FN),
                                                                          kwargs.get(WEIGHTS_QUANT_PARAMS_FN),
                                                                          kwargs.get(WEIGHTS_CHANNELS_AXIS))
