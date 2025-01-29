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
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_calculator import \
    ResourceUtilizationCalculator, BitwidthMode, TargetInclusionCriterion
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.target_platform_capabilities import FrameworkQuantizationCapabilities


def compute_resource_utilization_data(in_model: Any,
                                      representative_data_gen: Callable,
                                      core_config: CoreConfig,
                                      fqc: FrameworkQuantizationCapabilities,
                                      fw_info: FrameworkInfo,
                                      fw_impl: FrameworkImplementation,
                                      transformed_graph: Graph = None,
                                      mixed_precision_enable: bool = True) -> ResourceUtilization:
    """
    Compute Resource Utilization information that can be relevant for defining target ResourceUtilization for mixed precision search.
    Calculates maximal activation tensor size, the sum of the model's weight parameters and the total memory combining both weights
    and maximal activation tensor size.

    Args:
        in_model:  Model to build graph from (the model that intended to be quantized).
        representative_data_gen: Dataset used for calibration.
        core_config: CoreConfig containing parameters of how the model should be quantized.
        fqc: FrameworkQuantizationCapabilities object that models the inference target platform and
                                              the attached framework operator's information.
        fw_info: Information needed for quantization about the specific framework.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        transformed_graph: An internal graph representation of the input model. Defaults to None.
                            If no graph is provided, a graph will be constructed using the specified model.
        mixed_precision_enable: Indicates if mixed precision is enabled, defaults to True.
                                If disabled, computes resource utilization using base quantization
                                configurations across all layers.

    Returns:
        ResourceUtilization: An object encapsulating the calculated resource utilization computations.

    """
    core_config = _create_core_config_for_ru(core_config)
    # We assume that the resource_utilization_data API is used to compute the model resource utilization for
    # mixed precision scenario, so we run graph preparation under the assumption of enabled mixed precision.
    if transformed_graph is None:
        transformed_graph = graph_preparation_runner(in_model,
                                                     representative_data_gen,
                                                     core_config.quantization_config,
                                                     fw_info,
                                                     fw_impl,
                                                     fqc,
                                                     bit_width_config=core_config.bit_width_config,
                                                     mixed_precision_enable=mixed_precision_enable,
                                                     running_gptq=False)

    ru_calculator = ResourceUtilizationCalculator(transformed_graph, fw_impl, fw_info)
    ru = ru_calculator.compute_resource_utilization(TargetInclusionCriterion.AnyQuantized, BitwidthMode.Q8Bit,
                                                    ru_targets=set(RUTarget) - {RUTarget.BOPS})
    ru.bops, _ = ru_calculator.compute_bops(TargetInclusionCriterion.AnyQuantized, BitwidthMode.Float)
    return ru


def requires_mixed_precision(in_model: Any,
                             target_resource_utilization: ResourceUtilization,
                             representative_data_gen: Callable,
                             core_config: CoreConfig,
                             fqc: FrameworkQuantizationCapabilities,
                             fw_info: FrameworkInfo,
                             fw_impl: FrameworkImplementation) -> bool:
    """
    The function checks whether the model requires mixed precision to meet the requested target resource utilization.
    This is determined by whether the target memory usage of the weights is less than the available memory,
    the target maximum size of an activation tensor is less than the available memory,
    and the target number of BOPs is less than the available BOPs.
    If any of these conditions are met, the function returns True. Otherwise, it returns False.

    Args:
        in_model: The model to be evaluated.
        target_resource_utilization: The resource utilization of the target device.
        representative_data_gen: A function that generates representative data for the model.
        core_config: CoreConfig containing parameters of how the model should be quantized.
        fqc: FrameworkQuantizationCapabilities object that models the inference target platform and
                                              the attached framework operator's information.
        fw_info: Information needed for quantization about the specific framework.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns: A boolean indicating if mixed precision is needed.
    """
    # Any target resource utilization other than weights will always require MP calculation.
    if target_resource_utilization.activation_restricted() or \
            target_resource_utilization.total_mem_restricted() or \
            target_resource_utilization.bops_restricted():
        return True

    core_config = _create_core_config_for_ru(core_config)

    transformed_graph = graph_preparation_runner(in_model,
                                                 representative_data_gen,
                                                 core_config.quantization_config,
                                                 fw_info,
                                                 fw_impl,
                                                 fqc,
                                                 bit_width_config=core_config.bit_width_config,
                                                 mixed_precision_enable=False,
                                                 running_gptq=False)

    ru_calculator = ResourceUtilizationCalculator(transformed_graph, fw_impl, fw_info)
    max_ru = ru_calculator.compute_resource_utilization(TargetInclusionCriterion.AnyQuantized, BitwidthMode.QMaxBit,
                                                        ru_targets=target_resource_utilization.get_restricted_targets())
    return not target_resource_utilization.is_satisfied_by(max_ru)


def _create_core_config_for_ru(core_config: CoreConfig) -> CoreConfig:
    """
    Create a core config to use for resource utilization computation.

    Args:
        core_config: input core config

    Returns:
        Core config for resource utilization.
    """
    core_config = copy.deepcopy(core_config)
    # For resource utilization graph_preparation_runner runs with gptq=False (the default value). HMSE is not supported
    # without GPTQ and will raise an error later so we replace it with MSE.
    if core_config.quantization_config.weights_error_method == QuantizationErrorMethod.HMSE:
        core_config.quantization_config.weights_error_method = QuantizationErrorMethod.MSE
    return core_config
