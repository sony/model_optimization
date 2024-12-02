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
from typing import Callable, List, Tuple

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import BaseNodeQuantizationConfig, \
    NodeWeightsQuantizationConfig, NodeActivationQuantizationConfig
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import AttributeQuantizationConfig, \
    OpQuantizationConfig
from model_compression_toolkit.logger import Logger


##########################################
# Every node holds a quantization configuration
# for its weights quantization, and a different quantization
# configuration for its activation quantization configuration.
##########################################

class CandidateNodeQuantizationConfig(BaseNodeQuantizationConfig):
    """
    Class for representing candidate node configuration, which includes weights and activation configuration combined.
    """

    def __init__(self,
                 qc: QuantizationConfig = None,
                 op_cfg: OpQuantizationConfig = None,
                 activation_quantization_cfg: NodeActivationQuantizationConfig = None,
                 activation_quantization_fn: Callable = None,
                 activation_quantization_params_fn: Callable = None,
                 weights_quantization_cfg: NodeWeightsQuantizationConfig = None,
                 weights_channels_axis: Tuple[int, int] = None,
                 node_attrs_list: List[str] = None):
        """

        Args:
            qc: QuantizationConfig to create the node's config from.
            op_cfg: OpQuantizationConfig of the node with quantizers types to use when creating node quantization configuration.
            activation_quantization_cfg: An option to pass a NodeActivationQuantizationConfig to create a new config from.
            activation_quantization_fn: Function to use when quantizing the node's activations.
            activation_quantization_params_fn: Function to use when computing the threshold for quantizing a node's activations.
            weights_quantization_cfg: An option to pass a NodeWeightsQuantizationConfig to create a new config from.
            weights_channels_axis: Axis to quantize a node's weights attribute when quantizing per-channel.
            node_attrs_list: A list of the node's weights attributes names.
        """

        if activation_quantization_cfg is not None:
            self.activation_quantization_cfg = activation_quantization_cfg
        else:
            if any(v is None for v in (qc, op_cfg, activation_quantization_fn, activation_quantization_params_fn)):  # pragma: no cover
                Logger.critical(
                    "Missing required arguments to initialize a node activation quantization configuration. "
                    "Ensure QuantizationConfig, OpQuantizationConfig, activation quantization function, "
                    "and parameters function are provided.")
            self.activation_quantization_cfg = (
                NodeActivationQuantizationConfig(qc=qc,
                                                 op_cfg=op_cfg,
                                                 activation_quantization_fn=activation_quantization_fn,
                                                 activation_quantization_params_fn=activation_quantization_params_fn))

        if weights_quantization_cfg is not None:
            self.weights_quantization_cfg = weights_quantization_cfg
        else:
            if any(v is None for v in (qc, op_cfg, node_attrs_list)):  # pragma: no cover
                Logger.critical("Missing required arguments to initialize a node weights quantization configuration. "
                                "Ensure QuantizationConfig, OpQuantizationConfig, weights quantization function, "
                                "parameters function, and weights attribute quantization config are provided.")
            self.weights_quantization_cfg = NodeWeightsQuantizationConfig(qc=qc, op_cfg=op_cfg,
                                                                          weights_channels_axis=weights_channels_axis,
                                                                          node_attrs_list=node_attrs_list)
