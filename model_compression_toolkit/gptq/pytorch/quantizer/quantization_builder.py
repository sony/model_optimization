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
from typing import List, Dict, Tuple

from model_compression_toolkit.gptq import GradientPTQConfig
from model_compression_toolkit.core import common
from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.node_to_quantizer import \
    get_activation_inferable_quantizer_kwargs
from model_compression_toolkit.gptq.pytorch.quantizer.base_pytorch_gptq_quantizer import \
    BasePytorchGPTQTrainableQuantizer
from mct_quantizers import QuantizationTarget
from mct_quantizers.common.get_quantizers import get_inferable_quantizer_class
from mct_quantizers.pytorch.quantizers import BasePyTorchInferableQuantizer

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.trainable_infrastructure import TrainingMethod, BasePytorchActivationTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure.common.get_quantizer_config import \
    get_trainable_quantizer_weights_config, get_trainable_quantizer_activation_config
from model_compression_toolkit.trainable_infrastructure.common.get_quantizers import \
    get_trainable_quantizer_class


def quantization_builder(n: common.BaseNode,
                         gptq_config: GradientPTQConfig,
                         kernel_attr: str = None
                         ) -> Tuple[Dict[str, BasePytorchGPTQTrainableQuantizer], List[BasePyTorchInferableQuantizer]]:
    """
    Build quantizers for a node according to its quantization configuration and
    a global NoOpQuantizeConfig object.

    Args:
        n: Node to build its QuantizeConfig.
        gptq_config (GradientPTQConfig): GradientPTQConfig configuration.
        kernel_attr: A potential kernel attribute name to build its trainable quantizer.

    Returns:
        A dictionary which maps the weights kernel attribute to a quantizer for GPTQ training.
        Note that we return a dictionary although there is only a single attribute that is being mapped to a quantizer,
        to be compatible with the quantization infrastructure template.
    """

    weights_quantizers = {}
    if kernel_attr is not None and n.is_weights_quantization_enabled(kernel_attr):
        # Only nodes with kernel attribute are trainable during GPTQ
        quant_method = n.final_weights_quantization_cfg.get_attr_config(kernel_attr).weights_quantization_method
        quantizer_class = get_trainable_quantizer_class(quant_target=QuantizationTarget.Weights,
                                                        quantizer_id=gptq_config.rounding_type,
                                                        quant_method=quant_method,
                                                        quantizer_base_class=BasePytorchGPTQTrainableQuantizer)
        weights_quantizers.update({kernel_attr: quantizer_class(get_trainable_quantizer_weights_config(n,
                                                                                                       kernel_attr),
                                                                **gptq_config.gptq_quantizer_params_override)})
    activation_quantizers = []
    if n.is_activation_quantization_enabled():
        if n.final_activation_quantization_cfg is None:
            Logger.critical(f"Cannot set quantizer for a node without a final activation quantization configuration.")  # pragma: no cover

        quant_method = n.final_activation_quantization_cfg.activation_quantization_method

        quantizer_class = get_trainable_quantizer_class(quant_target=QuantizationTarget.Activation,
                                                        quantizer_id=TrainingMethod.STE,
                                                        quant_method=quant_method,
                                                        quantizer_base_class=BasePytorchActivationTrainableQuantizer)
        cfg = get_trainable_quantizer_activation_config(n, None)
        activation_quantizers.append(quantizer_class(cfg, freeze_quant_params=True))

    return weights_quantizers, activation_quantizers
