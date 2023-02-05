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
from typing import List, Dict, Tuple

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.qat.common.qat_config import QATConfig
from model_compression_toolkit.qat.common.qat_get_quantizer import get_quantizer_class
from model_compression_toolkit.qat.common.qat_get_quantizer_config import \
    get_trainable_quantizer_quantization_candidates, get_trainable_quantizer_weights_config, \
    get_trainable_quantizer_activation_config
from model_compression_toolkit.qat.pytorch.quantizer.base_pytorch_qat_quantizer import BasePytorchQATTrainableQuantizer
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget


def quantization_builder(n: common.BaseNode,
                         qat_config: QATConfig,
                         fw_info: FrameworkInfo,
                         ) -> Tuple[Dict[str, BasePytorchQATTrainableQuantizer],
                                    List[BasePytorchQATTrainableQuantizer]]:
    """
    Build quantizers for a node according to its quantization configuration and
    a global NoOpQuantizeConfig object.

    Args:
        n: Node to build its QuantizeConfig.
        qat_config (QATConfig): QAT configuration
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).

    Returns:
        weights_quantizers: A dictionary between a weight's name to its quantizer.
        activation_quantizers: A list of activations quantization, one for each layer output.).
    """
    if len(n.candidates_quantization_cfg) > 1:
        wq_cand, aq_cand = get_trainable_quantizer_quantization_candidates(n)
    else:
        wq_cand, aq_cand = None, None

    weight_quantizers = {}
    if n.is_weights_quantization_enabled():
        quant_method = n.final_weights_quantization_cfg.weights_quantization_method
        quantizer_class = get_quantizer_class(QuantizationTarget.Weights,
                                              qat_config.activation_training_method,
                                              quant_method,
                                              BasePytorchQATTrainableQuantizer)
        attributes = fw_info.get_kernel_op_attributes(n.type)
        for attr in attributes:
            weight_quantizers.update({attr: quantizer_class(get_trainable_quantizer_weights_config(n, wq_cand),
                                                           **qat_config.weight_quantizer_params_override)})

    activation_quantizers = []
    if n.is_activation_quantization_enabled():
        quant_method = n.final_activation_quantization_cfg.activation_quantization_method
        quantizer_class = get_quantizer_class(QuantizationTarget.Activation,
                                              qat_config.activation_training_method,
                                              quant_method,
                                              BasePytorchQATTrainableQuantizer)

        activation_quantizers = [quantizer_class(get_trainable_quantizer_activation_config(n, aq_cand),
                                                 **qat_config.activation_quantizer_params_override)]

    return weight_quantizers, activation_quantizers
