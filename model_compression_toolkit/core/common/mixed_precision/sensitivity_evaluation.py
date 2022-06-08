# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

from typing import Callable, List, Any

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation_manager import \
    SensitivityEvaluationManager, compute_mp_distance_measure


class SensitivityEvaluation:
    """
    Class to provide a function that evaluates the sensitivity of a bit-width configuration for the MP model.
    """

    def __init__(self,
                 graph: Graph,
                 quant_config: MixedPrecisionQuantizationConfigV2,
                 representative_data_gen: Callable,
                 fw_info: FrameworkInfo,
                 fw_impl: Any,
                 set_layer_to_bitwidth: Callable,
                 get_quant_node_name: Callable):
        """
            Create an object that allows to compute the sensitivity metric of an MP model (the sensitivity
            is computed based on the similarity of the interest points' outputs between the MP model
            and the float model).
            First, we initiate a SensitivityEvaluationManager that handles the components which are necessary for
            evaluating the sensitivity. It initializes an MP model (a model where layers that can be configured in
            different bit-widths) and a baseline model (a float model).
            Then, and based on the outputs of these two models (for some batches from the representative_data_gen),
            we build a function to measure the sensitivity of a change in a bit-width of a model's layer.

            Args:
                graph: Graph to get its sensitivity evaluation for changes in bit-widths for different nodes.
                quant_config: MixedPrecisionQuantizationConfig containing parameters of how the model should be quantized.
                representative_data_gen: Dataset used for getting batches for inference.
                fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
                fw_impl: FrameworkImplementation object with a specific framework methods implementation.
                set_layer_to_bitwidth: A fw-dependent function that allows to configure a configurable MP model
                    with a specific bit-width configuration.
                get_quant_node_name: A fw-dependent function that takes a node's name and outputs the node's name in a
                    quantized model (according to the fw conventions).

            """

        self.graph = graph
        self.quant_config = quant_config
        self.representative_data_gen = representative_data_gen
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.set_layer_to_bitwidth = set_layer_to_bitwidth
        self.get_quant_node_name = get_quant_node_name

        # Initiate a SensitivityEvaluationManager that contains all relevant objects for computing the distance metric.
        # The SensitivityEvaluationManager object initiates an MP model and a baseline model to be used for
        # comparison in the distance computation.
        # It generates and stores a set of image batches for evaluation.
        # It also runs and stores the baseline model's inference on the generated batches.
        self.sem = SensitivityEvaluationManager(self.graph, self.fw_info, self.quant_config,
                                                self.representative_data_gen, self.fw_impl)

        # Casting images tensors to the framework tensor type.
        self.sem.images_batches = list(map(lambda in_arr: self.fw_impl.to_tensor(in_arr), self.sem.images_batches))

        # Initiating baseline_tensors_list since it is not initiated in SensitivityEvaluationManager init.
        self.sem.init_baseline_tensors_list()

    def compute_metric(self,
                       mp_model_configuration: List[int],
                       node_idx: List[int] = None,
                       baseline_mp_configuration: List[int] = None) -> float:
        """
        Compute the sensitivity metric of the MP model for a given configuration (the sensitivity
        is computed based on the similarity of the interest points' outputs between the MP model
        and the float model).

        Args:
            mp_model_configuration: Bitwidth configuration to use to configure the MP model.
            node_idx: A list of nodes' indices to configure (instead of using the entire mp_model_configuration).
            baseline_mp_configuration: A mixed-precision configuration to set the model back to after modifying it to
                compute the metric for the given configuration.

        Returns:
            The sensitivity metric of the MP model for a given configuration.
        """

        # Configure MP model with the given configuration.
        self._configure_bitwidths_model(self.sem.model_mp,
                                        self.sem.sorted_configurable_nodes_names,
                                        mp_model_configuration,
                                        node_idx)

        # Compute the distance matrix
        distance_matrix = self.sem.build_distance_metrix()

        # Configure MP model back to the same configuration as the baseline model if baseline provided
        if baseline_mp_configuration is not None:
            self._configure_bitwidths_model(self.sem.model_mp,
                                            self.sem.sorted_configurable_nodes_names,
                                            baseline_mp_configuration,
                                            node_idx)

        return compute_mp_distance_measure(distance_matrix, self.quant_config.distance_weighting_method)

    def _configure_bitwidths_model(self,
                                   model_mp: Any,
                                   sorted_configurable_nodes_names: List[str],
                                   mp_model_configuration: List[int],
                                   node_idx: List[int]):
        """
        Configure a dynamic model (namely, model with layers that their weights and activation
        bit-width can be configured) using an MP model configuration mp_model_configuration.

        Args:
            model_mp: Dynamic model to configure.
            sorted_configurable_nodes_names: List of configurable nodes names sorted topology.
            mp_model_configuration: Configuration of bit-width indices to set to the model.
            node_idx: List of nodes' indices to configure (the rest layers are configured as the baseline model).
        """

        # Configure model
        # Note: Not all nodes in the graph are included in the MP model that is returned by the model builder.
        # Thus, the last configurable layer must be included in the interest points for evaluating the metric,
        # otherwise, not all configurable nodes will be considered throughout the MP optimization search (since
        # they will not affect the metric value).
        model_mp_layers_names = self.fw_impl.get_model_layers_names(model_mp)
        if node_idx is not None:  # configure specific layers in the mp model
            for node_idx_to_configure in node_idx:
                node_name = self.get_quant_node_name(sorted_configurable_nodes_names[node_idx_to_configure])
                if node_name in model_mp_layers_names:
                    current_layer = self.fw_impl.get_model_layer_by_name(model_mp, node_name)
                    self.set_layer_to_bitwidth(current_layer, mp_model_configuration[node_idx_to_configure])
                else:
                    raise Exception("The last configurable node is not included in the list of interest points for"
                                    "sensitivity evaluation metric for the mixed-precision search.")

        else:  # use the entire mp_model_configuration to configure the model
            for node_idx_to_configure, bitwidth_idx in enumerate(mp_model_configuration):
                node_name = self.get_quant_node_name(sorted_configurable_nodes_names[node_idx_to_configure])
                if node_name in model_mp_layers_names:
                    current_layer = self.fw_impl.get_model_layer_by_name(model_mp, node_name)
                    self.set_layer_to_bitwidth(current_layer, mp_model_configuration[node_idx_to_configure])
                else:
                    raise Exception("The last configurable node is not included in the list of interest points for"
                                    "sensitivity evaluation metric for the mixed-precision search.")
