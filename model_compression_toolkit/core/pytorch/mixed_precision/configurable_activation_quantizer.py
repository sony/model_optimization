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
import torch
import torch.nn as nn
from typing import Dict, List, Any
from mct_quantizers.pytorch.quantizers import BasePyTorchInferableQuantizer

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.mixed_precision.configurable_quant_id import ConfigurableQuantizerIdentifier
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget
from mct_quantizers import mark_quantizer


@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC,
                                     QuantizationMethod.UNIFORM, QuantizationMethod.LUT_POT_QUANTIZER],
                quantizer_type=ConfigurableQuantizerIdentifier.CONFIGURABLE_ID)
class ConfigurableActivationQuantizer(BasePyTorchInferableQuantizer):
    """
    Trainable symmetric quantizer to optimize the rounding of the quantized values using a soft quantization method.
    """

    def __init__(self,
                 node_q_cfg: List[CandidateNodeQuantizationConfig],
                 max_candidate_idx: int = 0):
        """
        Construct a Pytorch model that utilize a fake weight quantizer of soft-quantizer for symmetric quantizer.

        Args:
            quantization_config: Trainable weights quantizer config.
            quantization_parameter_learning (Bool): Whether to learn the threshold or not
        """

        super(ConfigurableActivationQuantizer, self).__init__()

        self.node_q_cfg = node_q_cfg
        self.active_quantization_config_index = max_candidate_idx  # initialize with first config as default

        for qc in self.node_q_cfg:
            if qc.activation_quantization_cfg.enable_activation_quantization != \
                   self.node_q_cfg[0].activation_quantization_cfg.enable_activation_quantization:
                Logger.error("Candidates with different activation enabled properties is currently not supported.")

        # Setting layer's activation
        self.activation_quantizers = self._get_activation_quantizers()
        self.active_quantization_config_index = max_candidate_idx

    def _get_activation_quantizers(self) -> List[Any]:
        """
        Builds a list of quantizers for each of the bitwidth candidates for activation quantization,
        to be stored and used during MP search.

        Returns: a list of activation quantizers - for each bitwidth and layer's attribute to be quantized.
        """
        activation_quantizers = []
        for index, qc in enumerate(self.node_q_cfg):
            q_activation = self.node_q_cfg[index].activation_quantization_cfg
            activation_quantizers.append(q_activation.quantize_node_output)

        return activation_quantizers

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
        Quantize a tensor.

        Args:
            inputs: Input tensor to quantize.

        Returns:
            quantized tensor
        """

        return self.activation_quantizers[self.active_quantization_config_index](inputs)

    def get_config(self) -> Dict[str, Any]:  # pragma: no cover
        """
        Returns: The ConfigurableActivationQuantizer configuration.
        """

        return {
            'node_q_cfg': self.node_q_cfg,
            'activation_quantizers': self.activation_quantizers,
            'active_quantization_config_index': self.active_quantization_config_index
        }
