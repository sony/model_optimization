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
import copy
from typing import Callable, Any

from model_compression_toolkit.core import FrameworkInfo, ResourceUtilization, CoreConfig, QuantizationErrorMethod
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_calculator import \
    ResourceUtilizationCalculator, BitwidthMode, TargetInclusionCriterion
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.target_platform_capabilities import FrameworkQuantizationCapabilities


def compute_resource_utilization_data(in_model: Any,
                                      representative_data_gen: Callable,
                                      core_config: CoreConfig,
                                      fqc: FrameworkQuantizationCapabilities,
                                      fw_info: FrameworkInfo,
                                      fw_impl: FrameworkImplementation) -> ResourceUtilization:
    """
    Compute Resource Utilization of a model with the default single precision quantization.
    This can serve as a basis for defining target Resource Utilization for mixed precision search.

    Args:
        in_model:  Model to build graph from (the model that intended to be quantized).
        representative_data_gen: Dataset used for calibration.
        core_config: CoreConfig containing parameters of how the model should be quantized.
        fqc: FrameworkQuantizationCapabilities object that models the inference target platform and
                                              the attached framework operator's information.
        fw_info: Information needed for quantization about the specific framework.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        ResourceUtilization: An object encapsulating the calculated resource utilization computations.

    """
    core_config = copy.deepcopy(core_config)
    # For resource utilization graph_preparation_runner runs with gptq=False (the default value). HMSE is not supported
    # without GPTQ and will raise an error later so we replace it with MSE.
    if core_config.quantization_config.weights_error_method == QuantizationErrorMethod.HMSE:
        core_config.quantization_config.weights_error_method = QuantizationErrorMethod.MSE

    transformed_graph = graph_preparation_runner(in_model,
                                                 representative_data_gen=representative_data_gen,
                                                 quantization_config=core_config.quantization_config,
                                                 fw_info=fw_info,
                                                 fw_impl=fw_impl,
                                                 fqc=fqc,
                                                 bit_width_config=core_config.bit_width_config,
                                                 mixed_precision_enable=False,
                                                 running_gptq=False)

    ru_calculator = ResourceUtilizationCalculator(transformed_graph, fw_impl, fw_info)
    return ru_calculator.compute_resource_utilization(TargetInclusionCriterion.AnyQuantized, BitwidthMode.QDefaultSP)
