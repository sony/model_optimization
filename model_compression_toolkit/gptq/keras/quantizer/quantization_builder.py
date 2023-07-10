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
from typing import Dict, List, Tuple

from model_compression_toolkit.gptq import GradientPTQConfigV2
from model_compression_toolkit.core import common
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.exporter.model_wrapper.keras.builder.node_to_quantizer import \
    get_inferable_quantizer_kwargs
from model_compression_toolkit.gptq.common.gptq_graph import get_kernel_attribute_name_for_gptq
from model_compression_toolkit.gptq.keras.quantizer.base_keras_gptq_quantizer import BaseKerasGPTQTrainableQuantizer
from mct_quantizers import QuantizationTarget
from mct_quantizers.common.get_quantizers import get_inferable_quantizer_class
from mct_quantizers.keras.quantizers import BaseKerasInferableQuantizer

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.trainable_infrastructure.common.get_quantizer_config import \
    get_trainable_quantizer_weights_config
from model_compression_toolkit.trainable_infrastructure.common.get_quantizers import \
    get_trainable_quantizer_class


def quantization_builder(n: common.BaseNode,
                         gptq_config: GradientPTQConfigV2
                         ) -> Tuple[Dict[str, BaseKerasGPTQTrainableQuantizer], List[BaseKerasInferableQuantizer]]:
    """
    Build quantizers for a node according to its quantization configuration and
    a global NoOpQuantizeConfig object.

    Args:
        n: Node to build its QuantizeConfig.
        gptq_config (GradientPTQConfigV2): GradientPTQConfigV2 configuration.

    Returns:
        A dictionary which maps the weights kernel attribute to a quantizer for GPTQ training.
        Note that we return a dictionary although there is only a single attribute that is being mapped to a quantizer,
        to be compatible with the quantization infrastructure template.
    """

    weights_quantizers = {}
    if n.is_weights_quantization_enabled():
        quant_method = n.final_weights_quantization_cfg.weights_quantization_method

        quantizer_class = get_trainable_quantizer_class(quant_target=QuantizationTarget.Weights,
                                                        quantizer_id=gptq_config.rounding_type,
                                                        quant_method=quant_method,
                                                        quantizer_base_class=BaseKerasGPTQTrainableQuantizer)
        kernel_attribute = get_kernel_attribute_name_for_gptq(layer_type=n.type,
                                                              fw_info=DEFAULT_KERAS_INFO)

        weights_quantizers.update({kernel_attribute: quantizer_class(get_trainable_quantizer_weights_config(n),
                                                                     **gptq_config.gptq_quantizer_params_override)})

    activation_quantizers = []
    if n.is_activation_quantization_enabled():
        if n.final_activation_quantization_cfg is None:
            Logger.critical(f'Can not set quantizer for a node with no final activation quantization configuration')  #
            # pragma: no cover

        quant_method = n.final_activation_quantization_cfg.activation_quantization_method

        quantizer_class = get_inferable_quantizer_class(quant_target=QuantizationTarget.Activation,
                                                        quant_method=quant_method,
                                                        quantizer_base_class=BaseKerasInferableQuantizer)

        kwargs = get_inferable_quantizer_kwargs(n.final_activation_quantization_cfg, QuantizationTarget.Activation)

        activation_quantizers.append(quantizer_class(**kwargs))

    return weights_quantizers, activation_quantizers
