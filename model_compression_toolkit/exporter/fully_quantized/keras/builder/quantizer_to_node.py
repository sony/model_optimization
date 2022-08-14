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

from typing import List

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.constants import THRESHOLD, SIGNED, RANGE_MIN, RANGE_MAX
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import calculate_delta
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.exporter.fully_quantized.keras.quantizers.fq_quantizer import FakeQuantQuantizer
from model_compression_toolkit.exporter.fully_quantized.keras.quantizers.weights_uniform_quantizer import \
    WeightsUniformQuantizer


# Supporting other quantizer types in the future
SUPPORTED_WEIGHT_QUANTIZER_TYPES = [QuantizationMethod.POWER_OF_TWO,
                                    QuantizationMethod.SYMMETRIC,
                                    QuantizationMethod.UNIFORM]

SUPPORTED_ACTIVATION_QUANTIZER_TYPES = [QuantizationMethod.POWER_OF_TWO,
                                        QuantizationMethod.SYMMETRIC,
                                        QuantizationMethod.UNIFORM]


def get_weights_quantizer_for_node(node: BaseNode, weights_attr :List[str]):

    assert node.final_weights_quantization_cfg is not None, f'Can not set quantizer for a node with no final ' \
                                                            f'weights quantization configuration'

    node_w_qc = node.final_weights_quantization_cfg
    weights_quantization_method = node_w_qc.weights_quantization_method
    assert weights_quantization_method in SUPPORTED_WEIGHT_QUANTIZER_TYPES, \
        f'Fully quantized models are now supported for {SUPPORTED_WEIGHT_QUANTIZER_TYPES} quantization methods, but node ' \
        f'has {weights_quantization_method} quantization method'

    if weights_quantization_method in [QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC]:
        # TODO: Add assertion for POT case that thresholds are POT
        min_range = -node_w_qc.weights_quantization_params.get(THRESHOLD)
        max_range = node_w_qc.weights_quantization_params.get(THRESHOLD) - calculate_delta(
            node_w_qc.weights_quantization_params.get(THRESHOLD),
            n_bits=node_w_qc.weights_n_bits,
            signed=True)
    elif weights_quantization_method in [QuantizationMethod.UNIFORM]:
        min_range = node_w_qc.weights_quantization_params.get(RANGE_MIN)
        max_range = node_w_qc.weights_quantization_params.get(RANGE_MAX)
    else:
        raise NotImplemented

    assert len(weights_attr )==1, f'Currently, we support only one quantized weight per layer'
    return WeightsUniformQuantizer(node_w_qc.weights_n_bits,
                                   min_range,
                                   max_range,
                                   node.get_weights_by_keys(weights_attr[0]))


def get_activations_quantizer_for_node(node: BaseNode):

    assert node.final_activation_quantization_cfg is not None, f'Can not set quantizer for a node with no final ' \
                                                               f'activation quantization configuration'

    node_act_qc = node.final_activation_quantization_cfg
    activation_quantization_method = node_act_qc.activation_quantization_method
    assert activation_quantization_method in SUPPORTED_ACTIVATION_QUANTIZER_TYPES, \
        f'Fully quantized models are now supported for {SUPPORTED_ACTIVATION_QUANTIZER_TYPES} quantization methods, but node ' \
        f'has {activation_quantization_method} quantization method'

    if activation_quantization_method in [QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC]:
        # TODO: Add assertion for POT case that thresholds are POT
        min_range = 0
        if node_act_qc.activation_quantization_params.get(SIGNED):
            min_range = -node_act_qc.activation_quantization_params.get(THRESHOLD)
        max_range = node_act_qc.activation_quantization_params.get(THRESHOLD) - calculate_delta(
            node_act_qc.activation_quantization_params.get(THRESHOLD),
            n_bits=node_act_qc.activation_n_bits,
            signed=node_act_qc.activation_quantization_params.get(SIGNED))
    else:
        raise NotImplemented

    return FakeQuantQuantizer(node_act_qc.activation_n_bits,
                              min_range,
                              max_range)
