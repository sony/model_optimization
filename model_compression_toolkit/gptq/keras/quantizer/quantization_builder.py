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

from model_compression_toolkit import GradientPTQConfig, RoundingType
from model_compression_toolkit.core import common
from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.gptq.keras.quantizer.base_keras_gptq_quantizer import BaseKerasGPTQTrainableQuantizer
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.common.get_quantizer_config import \
    get_trainable_quantizer_weights_config
from model_compression_toolkit.quantizers_infrastructure.common.get_trainable_quantizer import get_quantizer_class


def quantization_builder(n: common.BaseNode,
                         gptq_config: GradientPTQConfig) -> Dict[str, BaseKerasGPTQTrainableQuantizer]:
    """
    Build quantizers for a node according to its quantization configuration and
    a global NoOpQuantizeConfig object.

    Args:
        n: Node to build its QuantizeConfig.
        gptq_config (GradientPTQConfig): GradientPTQConfig configuration.

    Returns:
        A dictionary which maps the weights kernel attribute to a quantizer for GPTQ training.
        Note that we return a dictionary although there is only a single attribute that is being mapped to a quantizer,
        to be compatible with the quantization infrastructure template.
    """

    if n.is_weights_quantization_enabled():
        quant_method = n.final_weights_quantization_cfg.weights_quantization_method

        quantizer_class = get_quantizer_class(QuantizationTarget.Weights,
                                              gptq_config.rounding_type,
                                              quant_method,
                                              BaseKerasGPTQTrainableQuantizer)

        return {KERNEL: quantizer_class(get_trainable_quantizer_weights_config(n),
                                        **_get_extended_quantizer_parametes(gptq_config))}
    else:
        return {}


def _get_extended_quantizer_parametes(gptq_config: GradientPTQConfig) -> Dict[str, Any]:
    """
    Return a dictionary with a mapping to necessary additional parameters for initializing the GPTQ quantizer.

    Args:
        gptq_config: A GPTQ configuration.

    Returns: A dictionary with parameters for initializing a quantizer.

    """

    if gptq_config.rounding_type == RoundingType.SoftQuantizer:
        return {'n_batches': gptq_config.quantizer_config.n_batches,
                'quantization_parameter_learning': gptq_config.quantization_parameters_learning,
                'n_epochs': gptq_config.n_epochs}

    return {}
