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
from typing import Tuple, Dict, List

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.qat.common.qat_get_quantizer_config import get_trainable_quantizer_weights_config, \
    get_trainable_quantizer_activation_config, get_trainable_quantizer_quantization_candidates
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.qat.keras.quantizer.base_keras_qat_quantizer import BaseKerasQATTrainableQuantizer
from model_compression_toolkit.qat.common.qat_config import QATConfig, TrainingMethod
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.common.constants import QUANTIZATION_TARGET, \
    QUANTIZATION_METHOD, QUANTIZER_TYPE
from model_compression_toolkit.quantizers_infrastructure.common.get_all_subclasses import get_all_subclasses


def _get_quantizer_class(quant_target: QuantizationTarget,
                         training_method: TrainingMethod,
                         quant_method: QuantizationMethod) -> type:
    """
    Searches for a quantizer class that matches the requested QuantizationTarget and QuantizationMethod and TrainingMethod.
    Exactly one class should be found.

    Args:
        quant_target: QuantizationTarget value which indicates what is the target for quantization to
            use the quantizer for.
        quant_method: A list of QuantizationMethod values to indicate all type of quantization methods that the
            quantizer supports.

    Returns: A class of a quantizer that inherits from BaseKerasQATTrainableQuantizer.

    """
    qat_quantizer_classes = get_all_subclasses(BaseKerasQATTrainableQuantizer)
    filtered_quantizers = list(filter(lambda q_class: getattr(q_class, QUANTIZATION_TARGET) == quant_target and
                                                      getattr(q_class, QUANTIZATION_METHOD) is not None and
                                                       quant_method in getattr(q_class, QUANTIZATION_METHOD) and
                                                      getattr(q_class, QUANTIZER_TYPE) == training_method,
                                      qat_quantizer_classes))

    if len(filtered_quantizers) != 1:
        Logger.error(f"Found {len(filtered_quantizers)} quantizer for target {quant_target.value} "
                     f"that matches the requested quantization method {quant_method.name} and "
                     f"quantizer type {training_method.value} but there should be exactly one."
                     f"The possible quantizers that were found are {filtered_quantizers}.")

    return filtered_quantizers[0]


def quantization_builder(n: common.BaseNode,
                         qat_config: QATConfig,
                         fw_info: FrameworkInfo,
                         ) -> Tuple[Dict[str, BaseKerasQATTrainableQuantizer], List[BaseKerasQATTrainableQuantizer]]:
    """
    Build quantizers for a node according to its quantization configuration and
    a global NoOpQuantizeConfig object.

    Args:
        n: Node to build its QuantizeConfig.
        qat_config (QATConfig): QAT configuration
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).

    Returns:
        weights_quantizers: A dictionary between a weight's name to its quantizer.
        activation_quantizers: A list of activations quantization, one for each layer output.
    """
    if len(n.candidates_quantization_cfg) > 1:
        wq_cand, aq_cand = get_trainable_quantizer_quantization_candidates(n)
    else:
        wq_cand, aq_cand = None, None

    weight_quantizers = {}
    if n.is_weights_quantization_enabled():
        quant_method = n.final_weights_quantization_cfg.weights_quantization_method

        quantizer_class = _get_quantizer_class(QuantizationTarget.Weights,
                                               qat_config.weight_training_method,
                                               quant_method)
        attributes = fw_info.get_kernel_op_attributes(n.type)
        for attr in attributes:
            weight_quantizers.update({attr: quantizer_class(get_trainable_quantizer_weights_config(n, wq_cand),
                                                            **qat_config.weight_quantizer_params_override)})

    activation_quantizers = []
    if n.is_activation_quantization_enabled():
        _quant_method = n.final_activation_quantization_cfg.activation_quantization_method
        # single output -> normalize to list of output_shapes
        output_shapes = n.output_shape if isinstance(n.output_shape[0], (list, tuple)) else [n.output_shape]

        quantizer_class = _get_quantizer_class(QuantizationTarget.Activation,
                                               qat_config.activation_training_method,
                                               _quant_method)

        activation_quantizers = [quantizer_class(get_trainable_quantizer_activation_config(n, aq_cand),
                                                 **qat_config.activation_quantizer_params_override)] * len(output_shapes)

    return weight_quantizers, activation_quantizers
