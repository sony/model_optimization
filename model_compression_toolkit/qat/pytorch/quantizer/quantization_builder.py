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
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.qat.common.qat_config import QATConfig, TrainingMethod
from model_compression_toolkit.qat.pytorch.quantizer.base_pytorch_qat_quantizer import BasePytorchQATTrainableQuantizer
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.common.base_trainable_quantizer_config import \
    TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig
from model_compression_toolkit.quantizers_infrastructure.common.constants import QUANTIZATION_TARGET, \
    QUANTIZATION_METHOD, QUANTIZER_TYPE
from model_compression_toolkit.quantizers_infrastructure.common.get_all_subclasses import get_all_subclasses


# TODO: move the following "get...config" functions to be members of BaseNode


def get_trainable_quantizer_weights_config(n: common.BaseNode) -> TrainableQuantizerWeightsConfig:
    """
    Returns the relevant configurations for weights trainable quantizer

    Args:
        n: BaseNode

    Returns:
         TrainableQuantizerWeightsConfig - an object that contains the quantizer configuration
    """
    config = n.final_weights_quantization_cfg
    return TrainableQuantizerWeightsConfig(config.weights_quantization_method,
                                           config.weights_n_bits,
                                           config.weights_quantization_params,
                                           config.enable_weights_quantization,
                                           config.weights_channels_axis,
                                           config.weights_per_channel_threshold,
                                           config.min_threshold)


def get_trainable_quantizer_activation_config(n: common.BaseNode) -> TrainableQuantizerActivationConfig:
    """
    Returns configurations for activation trainable quantizer

    Args:
        n: BaseNode

    Returns:
         TrainableQuantizerActivationConfig - an object that contains the quantizer configuration
    """
    config = n.final_activation_quantization_cfg
    return TrainableQuantizerActivationConfig(config.activation_quantization_method,
                                              config.activation_n_bits,
                                              config.activation_quantization_params,
                                              config.enable_activation_quantization,
                                              config.min_threshold)


def _get_quantizer_class(quant_target: QuantizationTarget,
                         training_method: TrainingMethod,
                         quant_method: QuantizationMethod):
    """
    Searches for a quantizer class that matches the requested QuantizationTarget and QuantizationMethod and TrainingMethod.
    Exactly one class should be found.

    Args:
        quant_target: QuantizationTarget value which indicates what is the target for quantization to
            use the quantizer for.
        quant_method: A list of QuantizationMethod values to indicate all type of quantization methods that the
            quantizer supports.

    Returns: A class of a quantizer that inherits from BasePytorchQATTrainableQuantizer.

    """
    qat_quantizer_classes = get_all_subclasses(BasePytorchQATTrainableQuantizer)
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
    weight_quantizers = {}
    if n.is_weights_quantization_enabled():
        quant_method = n.final_weights_quantization_cfg.weights_quantization_method
        quantizer_class = _get_quantizer_class(QuantizationTarget.Weights,
                                               qat_config.activation_training_method,
                                               quant_method)
        attributes = fw_info.get_kernel_op_attributes(n.type)
        for attr in attributes:
            weight_quantizers.update({attr: quantizer_class(get_trainable_quantizer_weights_config(n),
                                                           **qat_config.weight_quantizer_params_override)})

    activation_quantizers = []
    if n.is_activation_quantization_enabled():
        quant_method = n.final_activation_quantization_cfg.activation_quantization_method
        quantizer_class = _get_quantizer_class(QuantizationTarget.Activation,
                                               qat_config.activation_training_method,
                                               quant_method)

        activation_quantizers = [quantizer_class(get_trainable_quantizer_activation_config(n),
                                                 **qat_config.activation_quantizer_params_override)]

    return weight_quantizers, activation_quantizers
