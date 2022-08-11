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
from model_compression_toolkit.core.common.quantization.node_quantization_config import BaseNodeNodeQuantizationConfig, \
    NodeWeightsQuantizationConfig, NodeActivationQuantizationConfig


##########################################
# Every node holds a quantization configuration
# for its weights quantization, and a different quantization
# configuration for its activation quantization configuration.
##########################################

class CandidateNodeQuantizationConfig(BaseNodeNodeQuantizationConfig):
    """
    Class for representing candidate node configuration, which includes weights and activation configuration combined.
    """

    def __init__(self, **kwargs):
        activation_quantization_cfg = kwargs.get('activation_quantization_cfg', None)
        if activation_quantization_cfg is not None:
            self.activation_quantization_cfg = activation_quantization_cfg
        else:
            self.activation_quantization_cfg = NodeActivationQuantizationConfig(kwargs.get('qc'),
                                                                                kwargs.get('op_cfg'),
                                                                                kwargs.get('activation_quantization_fn'),
                                                                                kwargs.get('activation_quantization_params_fn'))
        weights_quantization_cfg = kwargs.get('weights_quantization_cfg', None)
        if weights_quantization_cfg is not None:
            self.weights_quantization_cfg = weights_quantization_cfg
        else:
            self.weights_quantization_cfg = NodeWeightsQuantizationConfig(kwargs.get('qc'),
                                                                          kwargs.get('op_cfg'),
                                                                          kwargs.get('weights_quantization_fn'),
                                                                          kwargs.get('weights_quantization_params_fn'),
                                                                          kwargs.get('weights_channels_axis'))
