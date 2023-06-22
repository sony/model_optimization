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
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
from mct_quantizers.pytorch.quantizers import BasePyTorchInferableQuantizer

from model_compression_toolkit.core.common.mixed_precision.configurable_quant_id import ConfigurableQuantizerIdentifier
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget

from mct_quantizers import mark_quantizer


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC,
                                     QuantizationMethod.UNIFORM, QuantizationMethod.LUT_POT_QUANTIZER,
                                     QuantizationMethod.LUT_SYM_QUANTIZER],
                quantizer_type=ConfigurableQuantizerIdentifier.CONFIGURABLE_ID)
class ConfigurableWeightsQuantizer(BasePyTorchInferableQuantizer):
    """
    Trainable symmetric quantizer to optimize the rounding of the quantized values using a soft quantization method.
    """

    def __init__(self,
                 node_q_cfg,
                 float_weights,
                 max_candidate_idx):
        """
        Initializes a configurable quantizer.

        Args:
            node_q_cfg: Quantization configuration candidates of the node that generated the layer that will
                use this quantizer.
            float_weights: Float weights of the layer.
            max_candidate_idx: Index of the node's candidate that has the maximal bitwidth (must exist absolute max).
        """

        super(ConfigurableWeightsQuantizer, self).__init__()

        self.node_q_cfg = node_q_cfg
        self.float_weights = float_weights
        self.max_candidate_idx = max_candidate_idx

        # Make sure the candidates configurations arrived in descending order.
        curmax = (np.inf, np.inf)
        n_candidate_bits = [(x.weights_quantization_cfg.weights_n_bits, x.activation_quantization_cfg.activation_n_bits)
                            for x in self.node_q_cfg]
        for candidate_bits in n_candidate_bits:
            assert candidate_bits < curmax
            curmax = candidate_bits

        for qc in self.node_q_cfg:
            if qc.weights_quantization_cfg.enable_weights_quantization != \
                   self.node_q_cfg[0].weights_quantization_cfg.enable_weights_quantization:
                Logger.error("Candidates with different weights enabled properties is currently not supported.")

        # Setting the model with the initial quantized weights (the highest precision)
        self.weights_quantizer_fn_list = [qc.weights_quantization_cfg.weights_quantization_fn
                                          for qc in self.node_q_cfg]
        self.quantized_weights = self._get_quantized_weights()

        self.active_quantization_config_index = self.max_candidate_idx

    def _get_quantized_weights(self):
        """
        Calculates the quantized weights' tensors for each of the bitwidth candidates for quantization,
        to be stored and used during MP search.
        Returns: a list of quantized weights - for each bitwidth and layer's attribute to be quantized.
        """
        quantized_weights = []
        for index, qc in enumerate(self.node_q_cfg):
            # for each quantization configuration in mixed precision
            # get quantized weights for each attribute and for each filter
            q_weight = self.weights_quantizer_fn_list[index](tensor_data=self.float_weights,
                                                             n_bits=qc.weights_quantization_cfg.weights_n_bits,
                                                             signed=True,
                                                             quantization_params=qc.weights_quantization_cfg.weights_quantization_params,
                                                             per_channel=qc.weights_quantization_cfg.weights_per_channel_threshold,
                                                             output_channels_axis=qc.weights_quantization_cfg.weights_channels_axis)
            quantized_weights.append(to_torch_tensor(q_weight))

        return quantized_weights

    def set_weights_bit_width_index(self,
                                    index: int):
        """
        Change the "active" bitwidth index the configurable quantizer uses, so a different quantized weight
        will be used.

        Args:
            index: Quantization configuration candidate index to use.

        """

        assert index < len(self.node_q_cfg), \
            f'Quantizer has {len(self.node_q_cfg)} ' \
            f'possible nbits. Can not set index {index}'
        self.active_quantization_config_index = index

    def __call__(self,
                 inputs: nn.Parameter) -> torch.Tensor:
        """
        Method to return the quantized weight. This method is called when the framework needs to quantize a
            float weight, and is expected to return the quantized weight. Since we already quantized the weight in
            all possible bitwidths, we do not quantize it again, and simply return the quantized weight according
            to the current active_quantization_config_index.

        Args:
            inputs: Input tensor (not used in this function since the weights are already quantized).

        Returns:
            Quantized weight, that was quantized using number of bits that is in a
                specific quantization configuration candidate (the candidate's index is the
                index that is in active_quantization_config_index the quantizer holds).
        """

        return self.quantized_weights[self.active_quantization_config_index]

    def get_config(self) -> Dict[str, Any]:  # pragma: no cover
        """
        Returns: The ConfigurableWeightsQuantizer configuration.
        """

        return {
            'float_weights': self.float_weights,
            'node_q_cfg': self.node_q_cfg,
            'active_quantization_config_index': self.active_quantization_config_index
        }


