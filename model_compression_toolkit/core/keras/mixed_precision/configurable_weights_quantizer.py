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
from functools import partial
from typing import Dict, Any, List, Optional

from model_compression_toolkit.core.common.mixed_precision.configurable_quantizer_utils import \
    verify_candidates_descending_order, init_quantized_weights
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.logger import Logger
from mct_quantizers import QuantizationTarget, QuantizationMethod
from mct_quantizers import mark_quantizer

import tensorflow as tf
from model_compression_toolkit.core.common.mixed_precision.configurable_quant_id import \
    ConfigurableQuantizerIdentifier
from mct_quantizers.keras.quantizers import BaseKerasInferableQuantizer


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC,
                                     QuantizationMethod.UNIFORM, QuantizationMethod.LUT_POT_QUANTIZER,
                                     QuantizationMethod.LUT_SYM_QUANTIZER],
                identifier=ConfigurableQuantizerIdentifier.CONFIGURABLE_ID)
class ConfigurableWeightsQuantizer(BaseKerasInferableQuantizer):
    """
    Configurable weights quantizer for Keras mixed precision search.
    The quantizer holds a set of quantized layer's weights for each of the given bit-width candidates, provided by the
    node's quantization config. This allows to use different quantized weights on-the-fly.

    The general idea behind this kind of quantizer is that it gets the float tensor to quantize
    when initialized, it quantizes the float tensor in different bitwidths, and every time it need to return a
    quantized version of the float weight, it returns only one quantized weight according to an "active"
    index - the index of a candidate weight quantization configuration from a list of candidates that was passed
    to the quantizer when it was initialized.
    """

    def __init__(self,
                 node_q_cfg: List[CandidateNodeQuantizationConfig],
                 float_weights: tf.Tensor,
                 kernel_attr: str,
                 max_candidate_idx: int = 0):
        """
        Initializes a configurable quantizer.

        Args:
            node_q_cfg: Quantization configuration candidates of the node that generated the layer that will
                use this quantizer.
            float_weights: Float weights of the layer.
            kernel_attr: The kernel attribute name of the node. Only layers with kernel op can be configured.
            max_candidate_idx: Index of the node's candidate that has the maximal bitwidth (must exist absolute max).

        """

        super(ConfigurableWeightsQuantizer, self).__init__()

        self.node_q_cfg = node_q_cfg
        self.float_weights = float_weights
        self.max_candidate_idx = max_candidate_idx
        self.kernel_attr = kernel_attr

        verify_candidates_descending_order(self.node_q_cfg, kernel_attr)

        for qc in self.node_q_cfg:
            if qc.weights_quantization_cfg.get_attr_config(self.kernel_attr).enable_weights_quantization != \
                    self.node_q_cfg[0].weights_quantization_cfg.get_attr_config(self.kernel_attr).enable_weights_quantization:
                Logger.critical("Mixing candidates with varying weights quantization states (enabled/disabled) is not supported.")

        # Initialize quantized weights for each weight that should be quantized.
        self.quantized_weights = init_quantized_weights(node_q_cfg=self.node_q_cfg,
                                                        float_weights=self.float_weights,
                                                        fw_tensor_convert_func=partial(tf.convert_to_tensor,
                                                                                       dtype=tf.float32),
                                                        kernel_attr=self.kernel_attr)

        self.active_quantization_config_index = self.max_candidate_idx

    def set_weights_bit_width_index(self,
                                    index: Optional[int]):
        """
        Change the "active" bitwidth index the configurable quantizer uses, so a different quantized weight
        will be used.

        Args:
            index: Quantization configuration candidate index to use, or None to disable quantization.

        """

        if index and index >= len(self.node_q_cfg):
            Logger.critical(f'Quantizer supports only {len(self.node_q_cfg)} bit width configurations; index {index} is out of range.')# pragma: no cover
        self.active_quantization_config_index = index

    def __call__(self,
                 inputs: tf.Tensor) -> tf.Tensor:
        """
        Method to return the quantized weight. This method is called when the framework needs to quantize a
        float weight, and is expected to return the quantized weight. Since we already quantized the weight in
        all possible bitwidths, we do not quantize it again, and simply return the quantized weight according
        to the current active_quantization_config_index.

        Args:
            inputs: Input tensor (not relevant since the weights are already quantized).

        Returns:
            Quantized weight, that was quantized using number of bits that is in a
            specific quantization configuration candidate (the candidate's index is the
            index that is in active_quantization_config_index the quantizer holds).
        """
        if self.active_quantization_config_index is None:
            return self.float_weights
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
