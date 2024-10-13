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
from typing import Tuple, Dict, List, Callable

from model_compression_toolkit.core import common
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.qat.common.qat_config import QATConfig
from mct_quantizers import QuantizationTarget, KerasActivationQuantizationHolder
from model_compression_toolkit.qat.keras.quantizer.base_keras_qat_weight_quantizer import \
    BaseKerasQATWeightTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure.common.get_quantizer_config import \
    get_trainable_quantizer_weights_config, get_trainable_quantizer_activation_config, \
    get_trainable_quantizer_quantization_candidates
from model_compression_toolkit.trainable_infrastructure.common.get_quantizers import \
    get_trainable_quantizer_class
from model_compression_toolkit.trainable_infrastructure.keras.activation_quantizers import \
    BaseKerasActivationTrainableQuantizer


def get_activation_quantizer_holder(n: common.BaseNode,
                                    qat_config: QATConfig) -> Callable:
    """
    Retrieve a KerasActivationQuantizationHolder layer to use for activation quantization for a node.
    If the layer is not supposed to be wrapped with activation quantizers - return None.

    Args:
        n: Node to get KerasActivationQuantizationHolder to attach in its output.
        qat_config: Configuration of QAT (such as training methods for example).

    Returns:
        A KerasActivationQuantizationHolder layer for the node activation quantization.
    """
    _, activation_quantizers = quantization_builder(n,
                                                    qat_config)

    # Holder by definition uses a single quantizer for the activation quantization
    # thus we make sure this is the only possible case (unless it's a node with no activation
    # quantization, which in this case has an empty list).
    if len(activation_quantizers) == 1:
        return KerasActivationQuantizationHolder(activation_quantizers[0])
    Logger.critical(f'KerasActivationQuantizationHolder supports a single quantizer but {len(activation_quantizers)} quantizers were found for node {n}.')


def quantization_builder(n: common.BaseNode,
                         qat_config: QATConfig,
                         kernel_attr: str = None,
                         ) -> Tuple[Dict[str, BaseKerasQATWeightTrainableQuantizer], List[BaseKerasActivationTrainableQuantizer]]:
    """
    Build quantizers for a node according to its quantization configuration.

    Args:
        n: Node to build its QuantizeConfig.
        qat_config (QATConfig): QAT configuration
        kernel_attr: A potential kernel attribute name to build its trainable quantizer.


    Returns:
        weights_quantizers: A dictionary between a weight's name to its quantizer.
        activation_quantizers: A list of activations quantization, one for each layer output.
    """
    if len(n.candidates_quantization_cfg) > 1:
        wq_cand, aq_cand = get_trainable_quantizer_quantization_candidates(n, kernel_attr)
    else:
        wq_cand, aq_cand = None, None

    weight_quantizers = {}
    if kernel_attr is not None and n.is_weights_quantization_enabled(kernel_attr):
        # Only nodes with kernel attribute are trainable during QAT
        quant_method = n.final_weights_quantization_cfg.get_attr_config(kernel_attr).weights_quantization_method

        quantizer_class = get_trainable_quantizer_class(QuantizationTarget.Weights,
                                                        qat_config.weight_training_method,
                                                        quant_method,
                                                        BaseKerasQATWeightTrainableQuantizer)

        weight_quantizers.update({kernel_attr: quantizer_class(get_trainable_quantizer_weights_config(n,
                                                                                                      attr_name=kernel_attr,
                                                                                                      weights_quantization_candidates=wq_cand),
                                                        **qat_config.weight_quantizer_params_override)})

    activation_quantizers = []
    if n.is_activation_quantization_enabled():
        quant_method = n.final_activation_quantization_cfg.activation_quantization_method
        # single output -> normalize to list of output_shapes
        output_shapes = n.output_shape if isinstance(n.output_shape[0], (list, tuple)) else [n.output_shape]

        quantizer_class = get_trainable_quantizer_class(QuantizationTarget.Activation,
                                                        qat_config.activation_training_method,
                                                        quant_method,
                                                        BaseKerasActivationTrainableQuantizer)

        activation_quantizers = [quantizer_class(get_trainable_quantizer_activation_config(n, aq_cand),
                                                 **qat_config.activation_quantizer_params_override)] * len(output_shapes)

    return weight_quantizers, activation_quantizers
