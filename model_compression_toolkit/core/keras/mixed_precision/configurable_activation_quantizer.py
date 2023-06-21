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
from typing import List, Dict, Any

import numpy as np

from mct_quantizers.common.base_inferable_quantizer import mark_quantizer, QuantizationTarget
from mct_quantizers.common.constants import FOUND_TF
from mct_quantizers.common.quant_info import QuantizationMethod

from mct_quantizers.logger import Logger

from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.keras.mixed_precision.configurable_quant_id import ConfigurableQuantizerIdentifier

if FOUND_TF:
    import tensorflow as tf
    from mct_quantizers.keras.quantizers import BaseKerasInferableQuantizer

    @mark_quantizer(quantization_target=QuantizationTarget.Activation,
                    quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC,
                                         QuantizationMethod.UNIFORM, QuantizationMethod.LUT_POT_QUANTIZER,
                                         QuantizationMethod.LUT_SYM_QUANTIZER],
                    quantizer_type=ConfigurableQuantizerIdentifier.CONFIGURABLE_ID)
    class ConfigurableActivationQuantizer(BaseKerasInferableQuantizer):
        """
        Class for quantizing activations using power-of-two quantizer
        """

        def __init__(self,
                     node_q_cfg: List[CandidateNodeQuantizationConfig],
                     max_candidate_idx: int = 0):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                threshold: threshold for quantizing activations
                signed: whether or not to use signed quantization
            """
            # Call the superclass constructor with the given parameters, along with the target of Activation
            # quantization
            super(ConfigurableActivationQuantizer, self).__init__()

            # self.enable_activation_quantization = node_q_cfg[
            #     0].activation_quantization_cfg.enable_activation_quantization

            self.node_q_cfg = node_q_cfg
            self.active_quantization_config_index = max_candidate_idx  # initialize with first config as default

            self.activation_quantizers = []
            for qc in self.node_q_cfg:
                q_activation = qc.activation_quantization_cfg
                self.activation_quantizers.append(q_activation.quantize_node_output)

            # self.selective_quantizer = None if not self.enable_activation_quantization else \
            #     SelectiveActivationQuantizer(node_q_cfg, max_candidate_idx=max_candidate_idx)
            # self.selective_quantizer =

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

        def __call__(self,
                     inputs: tf.Tensor) -> np.ndarray:
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
            return self.activation_quantizers[self.active_quantization_config_index](inputs)

        def get_config(self) -> Dict[str, Any]:  # pragma: no cover
            """
            Returns: Configuration of TrainableQuantizer.
            """

            return {
                'node_q_cfg': self.node_q_cfg,
                'active_quantization_config_index': self.active_quantization_config_index,
                'activation_quantizers': self.activation_quantizers
            }

        # def __eq__(self, other: Any) -> bool:  # pragma: no cover
        #     """
        #     Check if equals to another object.
        #
        #     Args:
        #         other: Other object to compare.
        #
        #     Returns:
        #         Whether they are equal or not.
        #     """
        #     if not isinstance(other, SelectiveActivationQuantizer):
        #         return False
        #
        #     return self.node_q_cfg == other.node_q_cfg and \
        #            self.active_quantization_config_index == other.node_q_cfg and \
        #            self.activation_quantizers == other.activation_quantizers
        #
        # def __ne__(self, other: Any) -> bool:  # pragma: no cover
        #     """
        #     Check if not equals to another object.
        #
        #     Args:
        #         other: Other object to compare.
        #
        #     Returns:
        #         Whether they are differ or not.
        #     """
        #     return not self.__eq__(other)

else:
    class ConfigurableActivationQuantizer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            Logger.error('Installing tensorflow and tensorflow_model_optimization is mandatory '
                         'when using ConfigurableActivationQuantizer. '
                         'Could not find Tensorflow package.')
