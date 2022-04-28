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


from typing import Callable

from model_compression_toolkit.common.target_platform import OpQuantizationConfig
from model_compression_toolkit.common.quantization.node_quantization_config import BaseNodeNodeQuantizationConfig, \
    NodeWeightsQuantizationConfig, NodeActivationQuantizationConfig
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig


##########################################
# Every node holds a quantization configuration
# for its weights quantization, and a different quantization
# configuration for its activation quantization configuration.
##########################################

class CandidateNodeQuantizationConfig(BaseNodeNodeQuantizationConfig):
    """
    Class for representing candidate node configuration, which includes weights and activation configuration combined.
    """
    def __init__(self,
                 qc: QuantizationConfig,
                 op_cfg: OpQuantizationConfig,
                 activation_quantization_fn: Callable,
                 activation_quantization_params_fn: Callable,
                 weights_quantization_fn: Callable,
                 weights_quantization_params_fn: Callable,
                 weights_channels_axis: int
                 ):
        """

        Args:
            qc: QuantizationConfig to create the node's config from.
            op_cfg: OpQuantizationConfig of the node with quantizers types to use when creating node quantization configuration.
            activation_quantization_fn: Function to use when quantizing the node's activations.
            activation_quantization_params_fn: Function to use when computing the threshold for quantizing a node's activations.
            weights_quantization_fn: Function to use when quantizing the node's weights.
            weights_quantization_params_fn:  Function to use when computing the threshold for quantizing a node's weights.
            weights_channels_axis: Axis to quantize a node's kernel when quantizing per-channel.
        """

        self.activation_quantization_cfg = NodeActivationQuantizationConfig(qc,
                                                                            op_cfg,
                                                                            activation_quantization_fn,
                                                                            activation_quantization_params_fn)

        self.weights_quantization_cfg = NodeWeightsQuantizationConfig(qc,
                                                                      op_cfg,
                                                                      weights_quantization_fn,
                                                                      weights_quantization_params_fn,
                                                                      weights_channels_axis)
