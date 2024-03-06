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
from typing import Callable, Any

from tqdm import tqdm

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_activations_computation \
    import get_activations_qparams
from model_compression_toolkit.core.common.quantization.quantize_graph_weights import quantize_graph_weights
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute


def _collect_and_assign_act_threshold(graph: Graph,
                                      representative_data_gen: Callable,
                                      core_config: CoreConfig,
                                      fw_info: FrameworkInfo,
                                      fw_impl: FrameworkImplementation):
    """
    Collect statistics after second moment correction and assign new thresholds to activations.
     Args:
        graph: Graph to apply second moment correction.
        representative_data_gen (Callable): Dataset used for calibration.
        core_config (CoreConfig): Configuration object containing parameters of how the model should be
         quantized, including mixed precision parameters.
        fw_info: FrameworkInfo object with information about the specific framework's model.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
     """

    mi = ModelCollector(graph,
                        fw_impl,
                        fw_info,
                        core_config.quantization_config) # Mark points for statistics collection

    for _data in tqdm(representative_data_gen()):
        mi.infer(_data)

    for n in list(graph.nodes):
        if n.is_activation_quantization_enabled():
            activation_params = get_activations_qparams(
                activation_quant_cfg=n.final_activation_quantization_cfg,
                nodes_prior_info=n.prior_info,
                out_stats_container=graph.get_out_stats_collector(n))
            n.final_activation_quantization_cfg.set_activation_quantization_param(activation_params)


def quantized_model_builder_for_second_moment_correction(graph: common.Graph,
                                                         fw_info: FrameworkInfo,
                                                         fw_impl: Any):
    """
    Build a framework model from a graph for second moment correction.

    Args:
        graph: Graph to build the from.
        fw_info: FrameworkInfo object with information about the specific framework's model.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns:
        Quantized model for second moment correction.
    """
    quantized_tg = quantize_graph_weights(graph)

    quantized_model, user_info = fw_impl.model_builder(quantized_tg,
                                                       mode=ModelBuilderMode.FLOAT,
                                                       fw_info=fw_info)
    return quantized_model


def apply_second_moment_correction_to_graph(graph: Graph,
                                            representative_data_gen: Callable,
                                            core_config: CoreConfig,
                                            fw_info: FrameworkInfo,
                                            fw_impl: FrameworkImplementation) -> Graph:
    """
     Apply second moment correction on graph.
     Args:
        graph: Graph to apply second moment correction.
        representative_data_gen (Callable): Dataset used for calibration.
        core_config (CoreConfig): Configuration object containing parameters of how the model should be
         quantized, including mixed precision parameters.
        fw_info: FrameworkInfo object with information about the specific framework's model.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

     Returns:
         Graph after second moment correction.
     """
    semi_quantized_model = quantized_model_builder_for_second_moment_correction(graph, fw_info, fw_impl)
    fw_impl.apply_second_moment_correction(semi_quantized_model, core_config, representative_data_gen, graph)
    graph = substitute(graph, fw_impl.get_substitutions_after_second_moment_correction(core_config.quantization_config))
    _collect_and_assign_act_threshold(graph, representative_data_gen, core_config, fw_info, fw_impl)

    return graph
