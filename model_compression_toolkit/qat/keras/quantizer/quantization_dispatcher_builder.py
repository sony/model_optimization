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
from model_compression_toolkit import qunatizers_infrastructure as qi
from model_compression_toolkit.qat.keras.quantizer.ste_rounding.symmetirc_ste import STEWeightQuantizer
from model_compression_toolkit.qat.keras.quantizer.ste_rounding.uniform_ste import STEUniformWeightQuantizer

METHOD2QUANTIZER = {qi.QuantizationMethod.SYMMETRIC: STEWeightQuantizer,
                    qi.QuantizationMethod.POWER_OF_TWO: STEWeightQuantizer,
                    qi.QuantizationMethod.UNIFORM: STEUniformWeightQuantizer}


def quantization_dispatcher_builder(n: common.BaseNode,
                                    fw_info: FrameworkInfo,
                                    method2quantizer: Dict[
                                        qi.QuantizationMethod, qi.BaseKerasQuantizer] = METHOD2QUANTIZER) -> qi.KerasNodeQuantizationDispatcher:
    """
    Build a NodeQuantizationDispatcher for a node according to its quantization configuration and
    a global NoOpQuantizeConfig object.

    Args:
        n: Node to build its QuantizeConfig.
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
        method2quantizer: A mapping between quantization method to quantizer.

    Returns:
        A QuantizeConfig object with the appropriate quantizers (according to the node's
        quantization configuration).
    """
    nqd = qi.KerasNodeQuantizationDispatcher()
    if n.is_weights_quantization_enabled():
        attributes = fw_info.get_kernel_op_attributes(n.type)
        for attr in attributes:
            qunatizer_class = method2quantizer.get(n.final_weights_quantization_cfg.weights_quantization_method)
            if qunatizer_class is None:
                common.Logger.error(
                    f'Unknown Quantiztion method: {n.final_weights_quantization_cfg.weights_quantization_method}')
            nqd.add_weight_quantizer(attr, qunatizer_class(n.final_weights_quantization_cfg))

    return nqd
