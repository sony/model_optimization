# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Dict, List, Any

from model_compression_toolkit.core.common.mixed_precision.configurable_quant_id import ConfigurableQuantizerIdentifier
from model_compression_toolkit.core.common.mixed_precision.configurable_quantizer_utils import \
    verify_candidates_descending_order, init_activation_quantizers
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget
from mct_quantizers import mark_quantizer

import torch
import torch.nn as nn
from mct_quantizers.pytorch.quantizers import BasePyTorchInferableQuantizer


@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC,
                                     QuantizationMethod.UNIFORM, QuantizationMethod.LUT_POT_QUANTIZER],
                identifier=ConfigurableQuantizerIdentifier.CONFIGURABLE_ID)
class ConfigurableActivationQuantizer(BasePyTorchInferableQuantizer):
    """
    Configurable activation quantizer for mixed precision search.
    It holds a set of activation quantizers for each of the given bit-width candidates, provided by the
    node's quantization config. This allows to use different quantized activations on-the-fly, according to the
    "active" quantization configuration index.
    """

    def __init__(self,
                 node_q_cfg: List[CandidateNodeQuantizationConfig],
                 max_candidate_idx: int = 0):
        """
        Initializes a configurable quantizer.

        Args:
            node_q_cfg: Quantization configuration candidates of the node that generated the layer that will
                use this quantizer.
            max_candidate_idx: Index of the node's candidate that has the maximal bitwidth (must exist absolute max).
        """

        super(ConfigurableActivationQuantizer, self).__init__()

        self.node_q_cfg = node_q_cfg

        verify_candidates_descending_order(self.node_q_cfg)

        for qc in self.node_q_cfg:
            if qc.activation_quantization_cfg.enable_activation_quantization != \
                   self.node_q_cfg[0].activation_quantization_cfg.enable_activation_quantization:
                Logger.error("Candidates with different activation enabled properties is currently not supported.")  # pragma: no cover

        # Setting layer's activation
        self.activation_quantizers = init_activation_quantizers(self.node_q_cfg)
        self.active_quantization_config_index = max_candidate_idx  # initialize with first config as default

    def set_active_activation_quantizer(self,
                                        index: int):
        """
        Set an activation quantizer to use by the layer wrapped by the module.

        Args:
            index: Index of a candidate quantization configuration to use its quantizer
                for quantizing the activation.
        """

        assert index < len(self.node_q_cfg), f'Quantizer has {len(self.node_q_cfg)} ' \
                                             f'possible nbits. Can not set index {index}'
        self.active_quantization_config_index = index

    def __call__(self,
                 inputs: nn.Parameter) -> torch.Tensor:
        """
        Method to return the quantized activation tensor. This method is called when the framework needs to
        quantize a float activation tensor, and is expected to return the quantized tensor, according to the active
        activation quantizer.

        Args:
            inputs: Input tensor to quantize.

        Returns:
            Quantized activation tensor.
        """

        return self.activation_quantizers[self.active_quantization_config_index](inputs)
