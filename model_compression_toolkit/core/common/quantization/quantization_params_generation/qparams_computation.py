# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from tqdm import tqdm
from typing import List, Callable, Generator

from model_compression_toolkit.constants import NUM_QPARAM_HESSIAN_SAMPLES
from model_compression_toolkit.core import QuantizationErrorMethod
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_info import ChannelAxisMapping
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.hessian import HessianInfoService, HessianScoresRequest, HessianMode, \
    HessianScoresGranularity
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_activations_computation \
    import get_activations_qparams
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_weights_computation import \
    get_weights_qparams
from model_compression_toolkit.logger import Logger


def _collect_nodes_for_hmse(nodes_list: List[BaseNode], graph: Graph) -> List[BaseNode]:
    """
    Collects nodes that are compatiable for parameters selection search using HMSE,
    that is, have a kernel attribute that is configured for HMSE error method.

    Args:
        nodes_list: A list of nodes to search quantization parameters for.
        graph: Graph to compute its nodes' quantization parameters..

    Returns: A (possibly empty) list of nodes.

    """
    hmse_nodes = []
    for n in nodes_list:
        if n.kernel_attr is not None and n.is_weights_quantization_enabled(n.kernel_attr) and \
            all([c.weights_quantization_cfg.get_attr_config(n.kernel_attr).weights_error_method ==
                 QuantizationErrorMethod.HMSE for c in n.candidates_quantization_cfg]):
            hmse_nodes.append(n)

    return hmse_nodes


def calculate_quantization_params(graph: Graph,
                                  fw_impl: FrameworkImplementation,
                                  repr_data_gen_fn: Callable[[], Generator],
                                  nodes: List[BaseNode] = None,
                                  hessian_info_service: HessianInfoService = None,
                                  num_hessian_samples: int = NUM_QPARAM_HESSIAN_SAMPLES):
    """
    For a graph, go over its nodes, compute quantization params (for both weights and activations according
    to the given framework info), and create and attach a NodeQuantizationConfig to each node (containing the
    computed params).
    By default, the function goes over all nodes in the graph. However, specific nodes can be passed
    to compute quantization params only for them.

    Args:
        graph: Graph to compute its nodes' thresholds.
        fw_impl: FrameworkImplementation object.
        repr_data_gen_fn: callable returning representative dataset generator.
        nodes: List of nodes to compute their thresholds instead of computing it for all nodes in the graph.
        hessian_info_service: HessianInfoService object for retrieving Hessian-based scores (used only with HMSE error method).
        num_hessian_samples: Number of samples to approximate Hessian-based scores on (used only with HMSE error method).
    """

    Logger.info(f"\nRunning quantization parameters search. "
                f"This process might take some time, "
                f"depending on the model size and the selected quantization methods.\n")

    # Create a list of nodes to compute their thresholds
    nodes_list: List[BaseNode] = nodes or graph.nodes()

    # Collecting nodes that are configured to search weights quantization parameters using HMSE optimization
    # and computing required Hessian information to be used for HMSE parameters selection.
    # The Hessian scores are computed and stored in the hessian_info_service object.
    nodes_for_hmse = _collect_nodes_for_hmse(nodes_list, graph)
    if len(nodes_for_hmse) > 0:
        dataloader = fw_impl.convert_data_gen_to_dataloader(repr_data_gen_fn, batch_size=1)
        request = HessianScoresRequest(mode=HessianMode.WEIGHTS,
                                       granularity=HessianScoresGranularity.PER_ELEMENT,
                                       data_loader=dataloader,
                                       n_samples=num_hessian_samples,
                                       target_nodes=nodes_for_hmse)
        hessian_info_service.fetch_hessian(request)

    for n in tqdm(nodes_list, "Calculating quantization parameters"):  # iterate only nodes that we should compute their thresholds
        for candidate_qc in n.candidates_quantization_cfg:
            for attr in n.get_node_weights_attributes():
                if n.is_weights_quantization_enabled(attr):
                    # If the node's weights attribute should be quantized, we compute its quantization parameters
                    attr_cfg = candidate_qc.weights_quantization_cfg.get_attr_config(attr)
                    channels_axis = attr_cfg.weights_channels_axis
                    if channels_axis is not None:
                        output_channels_axis = channels_axis[0]
                    else:
                        output_channels_axis = None

                    mod_attr_cfg = attr_cfg

                    if attr_cfg.weights_error_method == QuantizationErrorMethod.HMSE:
                        # Although we collected nodes for HMSE before running the loop, we keep this verification to
                        # notify the user in case of HMSE configured for node that is not compatible for this method
                        if n.kernel_attr is None or n.kernel_attr not in attr:
                            Logger.warning(f"The HMSE error method for parameters selection is only supported for "
                                           f"kernel weights attributes. Running parameters selection for attribute "
                                           f"'{attr}' in node '{n.name}' with the default MSE error method instead.")
                            mod_attr_cfg = copy.deepcopy(attr_cfg)
                            mod_attr_cfg.weights_error_method = QuantizationErrorMethod.MSE

                    weights_params, output_channels_axis = get_weights_qparams(n.get_weights_by_keys(attr),
                                                                               candidate_qc.weights_quantization_cfg,
                                                                               mod_attr_cfg,
                                                                               output_channels_axis,
                                                                               node=n,
                                                                               hessian_info_service=hessian_info_service,
                                                                               num_hessian_samples=num_hessian_samples)
                    attr_cfg.weights_channels_axis = ChannelAxisMapping(output_channels_axis, attr_cfg.weights_channels_axis.input)
                    attr_cfg.set_weights_quantization_param(weights_params)

            if n.is_activation_quantization_enabled():
                # If node's activations should be quantized as well, we compute its activation quantization parameters
                activation_params = get_activations_qparams(
                    activation_quant_cfg=candidate_qc.activation_quantization_cfg,
                    nodes_prior_info=n.prior_info,
                    out_stats_container=graph.get_out_stats_collector(n))
                # Create a NodeQuantizationConfig containing all quantization params and attach it to the node
                candidate_qc.activation_quantization_cfg.set_activation_quantization_param(activation_params)
