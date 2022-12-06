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
import numpy as np

from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.quantization.quantization_analyzer import analyzer_graph
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.core.runner import get_finalized_graph, read_model_to_graph

from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_tp_model, \
    get_op_quantization_configs

import model_compression_toolkit as mct
tp = mct.target_platform


def prepare_graph_with_configs(in_model, fw_impl, fw_info, representative_dataset, get_tpc_func,
                               qc=DEFAULTCONFIG, mixed_precision_enabled=False):

    # TPC
    base_config, op_cfg_list = get_op_quantization_configs()

    # To override the default TP in the test - pass a TPC generator function that includes a generation of the TP
    # and doesn't use the TP that is passed from outside.
    _tp = generate_tp_model(base_config, base_config, op_cfg_list, "function_test")
    tpc = get_tpc_func("function_test", _tp)

    # Read Model
    graph = read_model_to_graph(in_model,
                                representative_data_gen=representative_dataset,
                                tpc=tpc,
                                fw_info=fw_info,
                                fw_impl=fw_impl)

    # Finalize graph with quantization configs
    graph = get_finalized_graph(graph,
                                tpc=tpc,
                                quant_config=qc,
                                fw_info=fw_info,
                                fw_impl=fw_impl,
                                mixed_precision_enable=mixed_precision_enabled)

    return graph


def prepare_graph_with_quantization_parameters(in_model, fw_impl, fw_info, representative_dataset, get_tpc_func,
                                               input_shape, qc=DEFAULTCONFIG, mixed_precision_enabled=False):
    graph = prepare_graph_with_configs(in_model, fw_impl, fw_info, representative_dataset, get_tpc_func,
                                       qc, mixed_precision_enabled)

    analyzer_graph(node_analyze_func=fw_impl.attach_sc_to_node,
                   graph=graph,
                   fw_info=fw_info,
                   qc=qc)

    mi = ModelCollector(graph,
                        fw_impl=fw_impl,
                        fw_info=fw_info)

    for i in range(10):
        mi.infer([np.random.randn(*input_shape)])

    calculate_quantization_params(graph,
                                  fw_info=fw_info,
                                  fw_impl=fw_impl)

    return graph