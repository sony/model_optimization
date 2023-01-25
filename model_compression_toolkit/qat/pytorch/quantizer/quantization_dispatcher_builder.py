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
from typing import Dict
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit import quantizers_infrastructure as qi
from model_compression_toolkit.qat.pytorch.quantizer.ste_rounding.symmetric_ste import STEWeightQuantizer, STEActivationQuantizer
from model_compression_toolkit.qat.pytorch.quantizer.ste_rounding.uniform_ste import STEUniformWeightQuantizer, STEUniformActivationQuantizer
from model_compression_toolkit.qat.common.qat_config import QATConfig, TrainingMethod
from model_compression_toolkit.quantizers_infrastructure.common.base_trainable_quantizer_config import \
    TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig

METHOD2WEIGHTQUANTIZER = {TrainingMethod.STE:{qi.QuantizationMethod.SYMMETRIC: STEWeightQuantizer,
                                              qi.QuantizationMethod.POWER_OF_TWO: STEWeightQuantizer,
                                              qi.QuantizationMethod.UNIFORM: STEUniformWeightQuantizer}}


METHOD2ACTQUANTIZER = {TrainingMethod.STE:{qi.QuantizationMethod.SYMMETRIC: STEActivationQuantizer,
                                           qi.QuantizationMethod.POWER_OF_TWO: STEActivationQuantizer,
                                           qi.QuantizationMethod.UNIFORM: STEUniformActivationQuantizer}}


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


def quantization_dispatcher_builder(n: common.BaseNode,
                                    qat_config: QATConfig,
                                    fw_info: FrameworkInfo,
                                    method2weightquantizer: Dict[
                                        qi.QuantizationMethod, qi.BasePytorchTrainableQuantizer] = None,
                                    method2actquantizer: Dict[
                                        qi.QuantizationMethod, qi.BasePytorchTrainableQuantizer] = None
                                    ) -> qi.PytorchNodeQuantizationDispatcher:

    """
    Build a NodeQuantizationDispatcher for a node according to its quantization configuration and
    a global NoOpQuantizeConfig object.

    Args:
        n: Node to build its QuantizeConfig.
        qat_config (QATConfig): QAT configuration
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
        method2weightquantizer: A mapping between quantization method to weight quantizer.
        method2actquantizer: A mapping between quantization method to activation quantizer.

    Returns:
        A QuantizeConfig object with the appropriate quantizers (according to the node's
        quantization configuration).
    """
    if method2weightquantizer is None:
        method2weightquantizer = METHOD2WEIGHTQUANTIZER
    if method2actquantizer is None:
        method2actquantizer = METHOD2ACTQUANTIZER

    nqd = qi.PytorchNodeQuantizationDispatcher()
    if n.is_weights_quantization_enabled():
        _quant_method = n.final_weights_quantization_cfg.weights_quantization_method
        if qat_config.weight_training_method not in method2weightquantizer:
            common.Logger.error(f'Unknown weight quantization training method: {_quant_method}')
        if _quant_method not in method2weightquantizer[qat_config.weight_training_method]:
            common.Logger.error(f'Unknown weight quantization method: {_quant_method}')
        quantizer_class = method2weightquantizer[qat_config.weight_training_method][_quant_method]
        attributes = fw_info.get_kernel_op_attributes(n.type)
        for attr in attributes:
            nqd.add_weight_quantizer(attr, quantizer_class(get_trainable_quantizer_weights_config(n), **qat_config.weight_quantizer_params_override))

    if n.is_activation_quantization_enabled():
        _quant_method = n.final_activation_quantization_cfg.activation_quantization_method
        if qat_config.activation_training_method not in method2actquantizer:
            common.Logger.error(f'Unknown activation quantization training method: {_quant_method}')
        if _quant_method not in method2actquantizer[qat_config.activation_training_method]:
            common.Logger.error(f'Unknown activation quantization method: {_quant_method}')
        quantizer_class = method2actquantizer[qat_config.activation_training_method][_quant_method]
        nqd.activation_quantizers = [quantizer_class(get_trainable_quantizer_activation_config(n), **qat_config.activation_quantizer_params_override)]

    return nqd
