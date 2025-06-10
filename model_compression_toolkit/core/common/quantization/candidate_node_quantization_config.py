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
from dataclasses import dataclass, InitVar
from typing import Callable, List, Tuple

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common.framework_info import ChannelAxisMapping
from model_compression_toolkit.core.common.quantization.node_quantization_config import BaseNodeQuantizationConfig, \
    NodeWeightsQuantizationConfig, NodeActivationQuantizationConfig, ActivationQuantizationMode
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OpQuantizationConfig
from model_compression_toolkit.logger import Logger


##########################################
# Every node holds a quantization configuration
# for its weights quantization, and a different quantization
# configuration for its activation quantization configuration.
##########################################

@dataclass
class TPCQuantizationInfo:
    # quantization config for single precision
    base_quantization_cfg: 'CandidateNodeQuantizationConfig'
    # quantization candidate configs for mixed precision
    candidates_quantization_cfg: List['CandidateNodeQuantizationConfig']

    validate: InitVar = True

    def __post_init__(self, validate=True):
        if validate and not any(self.base_quantization_cfg == qc for qc in self.candidates_quantization_cfg):
            raise ValueError('Candidates should contain the base config.')
        # TODO irena
        # for now make sure they are separate objects so that one doesnt inadvertently modify the other
        if any(self.base_quantization_cfg is qc for qc in self.candidates_quantization_cfg):
            self.base_quantization_cfg = copy.deepcopy(self.base_quantization_cfg)

    # TODO irena frozen / context manager?
    def update_activation_quantization_mode(self, mode: ActivationQuantizationMode):
        for c in self.candidates_quantization_cfg:
            c.activation_quantization_cfg.quant_mode = mode
        if self.base_quantization_cfg:
            self.base_quantization_cfg.activation_quantization_cfg.quant_mode = mode

    def disable_weights_quantization(self):
        for c in self.candidates_quantization_cfg:
            c.weights_quantization_cfg.enable_weights_quantization = False
        if self.base_quantization_cfg:
            self.base_quantization_cfg.weights_quantization_cfg.enable_weights_quantization = False


class CandidateNodeQuantizationConfig(BaseNodeQuantizationConfig):
    """
    Class for representing candidate node configuration, which includes weights and activation configuration combined.
    """

    def __init__(self,
                 op_cfg: OpQuantizationConfig = None,
                 activation_quantization_cfg: NodeActivationQuantizationConfig = None,
                 activation_quantization_fn: Callable = None,
                 activation_quantization_params_fn: Callable = None,
                 weights_quantization_cfg: NodeWeightsQuantizationConfig = None,
                 weights_channels_axis: ChannelAxisMapping = None,
                 node_attrs_list: List[str] = None):
        """

        Args:
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
            if any(v is None for v in (op_cfg, activation_quantization_fn, activation_quantization_params_fn)):  # pragma: no cover
                Logger.critical(
                    "Missing required arguments to initialize a node activation quantization configuration. "
                    "Ensure QuantizationConfig, OpQuantizationConfig, activation quantization function, "
                    "and parameters function are provided.")
            self.activation_quantization_cfg = (
                NodeActivationQuantizationConfig(op_cfg=op_cfg,
                                                 activation_quantization_fn=activation_quantization_fn,
                                                 activation_quantization_params_fn=activation_quantization_params_fn))

        if weights_quantization_cfg is not None:
            self.weights_quantization_cfg = weights_quantization_cfg
        elif all(v is not None for v in (op_cfg, node_attrs_list)):
            self.weights_quantization_cfg = NodeWeightsQuantizationConfig(op_cfg=op_cfg,
                                                                          weights_channels_axis=weights_channels_axis,
                                                                          node_attrs_list=node_attrs_list)
        else:
            self.weights_quantization_cfg = None
            Logger.debug("Setting weights quantization config as None during CandidateNodeQuantizationConfig creation."
                         "Notice, this should happen only for FLN nodes.")

    def __eq__(self, other):
        return (self.activation_quantization_cfg == other.activation_quantization_cfg and
                self.weights_quantization_cfg == other.weights_quantization_cfg)
