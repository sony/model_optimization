# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer
from typing import Dict, Any, List

from model_compression_toolkit.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig


class SelectiveWeightsQuantizer(Quantizer):
    """
    Quantizer that can use different quantized weights on-the-fly.
    The general idea behind this kind of quantizer is that it gets the float tensor to quantize
    when initialize, it quantizes the float tensor in different bitwidths, and every time it need to return a
    quantized version of the float weight, it returns only one quantized weight according to an "active"
    index - the index of a candidate weight quantization configuration from a list of candidates that was passed
    to the SelectiveWeightsQuantizer when it was initialized.
    The "active" index can be configured as part of the SelectiveWeightsQuantizer's API, so a different quantized
    weight can be returned in another time.
    """

    def __init__(self,
                 node_q_cfg: List[CandidateNodeQuantizationConfig],
                 float_weight: np.ndarray,
                 max_candidate_idx: int):
        """
        Init a selective quantizer.

        Args:
            node_q_cfg: Quantization configuration candidates of the node that generated the layer that will
                use this quantizer.
            float_weight: Float weights of the layer.
            max_candidate_idx: Index of the node's candidate that has the maximal bitwidth (must exist absolute max).
        """

        self.node_q_cfg = node_q_cfg
        self.quantizer_fn_list = [qc.weights_quantization_cfg.weights_quantization_fn for qc in self.node_q_cfg]
        self.float_weight = float_weight
        self.quantized_weights = []
        self.active_quantization_config_index = max_candidate_idx
        self._store_quantized_weights()

    def _quantize_by_qc(self, index: int) -> np.ndarray:
        """
        Quantize the quantizer float weight using a candidate quantization configuration.

        Args:
            index: Index of the candidate to use for the quantization.

        Returns:
            Quantized weight.
        """
        qc = self.node_q_cfg[index].weights_quantization_cfg
        return self.quantizer_fn_list[index](self.float_weight,
                                             qc.weights_n_bits,
                                             True,
                                             qc.weights_quantization_params,
                                             qc.weights_per_channel_threshold,
                                             qc.weights_channels_axis)

    def _store_quantized_weights(self):
        """

        Go over all candidates configurations, quantize the quantizer float weight according to each one
        of them, and store the quantized weights in a list quantized_weights the quantizer holds.

        """
        for i in range(len(self.node_q_cfg)):
            q_weight = self._quantize_by_qc(i)
            self.quantized_weights.append(tf.Variable(q_weight,
                                                      trainable=False,
                                                      dtype=tf.float32))

    def build(self,
              tensor_shape: TensorShape,
              name: str,
              layer: QuantizeWrapper) -> Dict[str, tf.Variable]:
        """
        The build method has to be implemented as part of the Keras framework,
        but there is no need to use it here as we do not train any new variable.
        Hence, it returns an empty dictionary.

        """

        return {}

    def __call__(self, inputs: tf.Tensor,
                 training: bool,
                 weights: Dict[str, tf.Variable],
                 **kwargs: Dict[str, Any]) -> np.ndarray:
        """
        Method to return the quantized weight. This method is called
        when the framework needs to quantize a float weight, and is expected to return the quantized
        weight. Since we already quantized the weight in all possible bitwidths, we do not
        quantize it again, and simply return the quantized weight according to the current
        active_quantization_config_index.

        Returns:
            Quantized weight, that was quantized using number of bits that is in a
            specific quantization configuration candidate (the candidate's index is the
            index that is in active_quantization_config_index the quantizer holds).
        """

        return self.quantized_weights[self.active_quantization_config_index]

    def set_active_quantization_config_index(self, index: int):
        """
        Set an index to use for the quantized weight the quantizer returns
        when requested.

        Args:
            index: Index of a candidate quantization configuration to use its quantized
            version of the float weight.

        """
        assert index < len(
            self.node_q_cfg), f'Quantizer has {len(self.node_q_cfg)} ' \
                                      f'possible nbits. Can not set ' \
                                      f'index {index}'
        self.active_quantization_config_index = index

    def get_active_quantization_config_index(self) -> int:
        """

        Returns: The index the quantizer uses when is selects which quantized weight
        to use when asked to (in __call__).

        """
        return self.active_quantization_config_index

    def get_active_quantized_weight(self) -> np.ndarray:
        """

        Returns: The current active quantized weight the quantizer holds.

        """
        return self.quantized_weights[self.active_quantization_config_index]

    def get_config(self) -> Dict[str, Any]:
        """
        Returns: Configuration of TrainableQuantizer.
        """

        return {
            'node_q_cfg': self.node_q_cfg,
            'float_weight': self.float_weight,
            'quantizer_fn_list': self.quantizer_fn_list
        }

    def __eq__(self, other: Any) -> bool:
        """
        Check if equals to another object.
        Args:
            other: Other object to compare.

        Returns:
            Whether they are equal or not.
        """
        if not isinstance(other, SelectiveWeightsQuantizer):
            return False

        return (self.node_q_cfg == other.node_q_cfg and
                self.float_weight == other.float_weight and
                self.quantizer_fn_list == other.quantizer_fn_list)

    def __ne__(self, other: Any) -> bool:
        """
        Check if not equals to another object.
        Args:
            other: Other object to compare.

        Returns:
            Whether they are differ or not.
        """
        return not self.__eq__(other)
