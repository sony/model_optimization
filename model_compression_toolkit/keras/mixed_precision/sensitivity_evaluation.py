# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

from tensorflow.python.layers.base import Layer
from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from typing import Callable, List, Any

from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common import BaseNode
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.keras.back2framework.model_builder import model_builder
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.keras.quantizer.mixed_precision.selective_weights_quantize_config import \
    SelectiveWeightsQuantizeConfig
import numpy as np


def get_sensitivity_evaluation(graph: Graph,
                               quant_config: MixedPrecisionQuantizationConfig,
                               metrics_weights_fn: Callable,
                               representative_data_gen: Callable,
                               fw_info: FrameworkInfo) -> Callable:
    """
    Create a function to compute the sensitivity metric of an MP model (the sensitivity
    is computed based on the similarity of the interest points' outputs between the MP model
    and the float model).
    First, we build a MP model (a model where layers that can be configured in different bitwidths use
    a SelectiveWeightsQuantizeConfig) and a baseline model (a float model).
    Then, and based on the outputs of these two models (for some batches from the representative_data_gen),
    we build a function to measure the sensitivity of a change in a bitwidth of a model's layer.

    Args:
        graph: Graph to get its sensitivity evaluation for changes in bitwidths for different nodes.
        quant_config: MixedPrecisionQuantizationConfig containing parameters of how the model should be quantized.
        metrics_weights_fn: Function to compute weights for a weighted average over the distances (per layer).
        representative_data_gen: Dataset used for getting batches for inference.
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).

    Returns:
        Function to compute the sensitivity metric.

    """

    interest_points = []  # List of graph nodes, the model should output their outputs.
    sorted_nodes = graph.get_configurable_sorted_nodes()
    sorted_configurable_nodes_names = []
    for n in sorted_nodes:
        interest_points.append(n)
        sorted_configurable_nodes_names.append(n.name)

    # Build a mixed-precision model which can be configured to use different bitwidth in different layers.
    model_mp, _ = model_builder(graph,
                                mode=ModelBuilderMode.MIXEDPRECISION,
                                append2output=interest_points,
                                fw_info=fw_info)

    # Build a baseline model.
    baseline_model = _build_baseline_model(graph,
                                           interest_points)


    def _compute_metric(mp_model_configuration: List[int],
                        node_idx: List[int] = None) -> float:
        """
        Compute the sensitivity metric of the MP model for a given configuration (the sensitivity
        is computed based on the similarity of the interest points' outputs between the MP model
        and the float model).

        Args:
            mp_model_configuration: Bitwidth configuration to use to configure the MP model.
            node_idx: A list of nodes' indices to configure (instead of using the entire mp_mp_model_configuration).

        Returns:
            The sensitivity metric of the MP model for a given configuration.
        """

        # Configure MP model with the given configuration.
        _configure_bitwidths_keras_model(model_mp,
                                         sorted_configurable_nodes_names,
                                         mp_model_configuration,
                                         node_idx)

        samples_count = 0  # Numer of images we used so far to compute the distance matrix.

        # List of distance matrices. We create a distance matrix for each sample from the representative_data_gen
        # and merge all of them eventually.
        distance_matrices = []

        # Compute the distance matrix for num_of_images images.
        while samples_count < quant_config.num_of_images:
            # Get a batch of images to infer in both models.
            inference_batch_input = representative_data_gen()
            batch_size = inference_batch_input[0].shape[0]

            # If we sampled more images than we should use in the distance matrix,
            # we take only a subset of these images and use only them for computing the distance matrix.
            if batch_size > quant_config.num_of_images-samples_count:
                inference_batch_input = [x[:quant_config.num_of_images-samples_count] for x in inference_batch_input]
                assert quant_config.num_of_images-samples_count == inference_batch_input[0].shape[0]
                batch_size = quant_config.num_of_images-samples_count

            samples_count += batch_size
            # If the model contains only one output we save it a list. If it's a list already, we keep it as a list.
            baseline_tensors = _tensors_as_list(baseline_model(inference_batch_input))

            # when using model.predict(), it does not uses the QuantizeWrapper functionality
            mp_tensors = _tensors_as_list(model_mp(inference_batch_input))

            # Build distance matrix: similarity between the baseline model to the float model
            # in every interest point for every image in the batch.
            distance_matrices.append(_build_distance_matrix(baseline_tensors,
                                                            mp_tensors,
                                                            quant_config.compute_distance_fn))

        # Merge all distance matrices into a single distance matrix.
        distance_matrix = np.concatenate(distance_matrices, axis=1)

        # Assert we used a correct number of images for computing the distance matrix
        assert distance_matrix.shape[1] == quant_config.num_of_images

        # Configure MP model back to the same configuration as the baseline model
        baseline_mp_configuration = [0] * len(mp_model_configuration)
        _configure_bitwidths_keras_model(model_mp,
                                         sorted_configurable_nodes_names,
                                         baseline_mp_configuration,
                                         node_idx)

        # Compute the distance between the baseline model's outputs and the MP model's outputs.
        # The distance is the mean of distances over all images in the batch that was inferred.
        mean_distance_per_layer = distance_matrix.mean(axis=1)
        # Use weights such that every layer's distance is weighted differently (possibly).
        return np.average(mean_distance_per_layer, weights=metrics_weights_fn(distance_matrix))

    return _compute_metric


def _configure_bitwidths_keras_model(model_mp: Model,
                                     sorted_configurable_nodes_names: List[str],
                                     mp_model_configuration: List[int],
                                     node_idx: List[int]):
    """
    Configure a dynamic Keras model (namely, model with layers that their weights
    bitwidth can be configured using SelectiveWeightsQuantizeConfig) using a MP
    model configuration mp_model_configuration.

    Args:
        model_mp: Dynamic Keras model to configure.
        sorted_configurable_nodes_names: List of configurable nodes names sorted topology.
        mp_model_configuration: Configuration of bitwidth indices to set to the model.
        node_idx: List of nodes' indices to configure (the rest layers are configured as the baseline model).

    """
    # Configure model
    if node_idx is not None:  # configure specific layers in the mp model
        for node_idx_to_configure in node_idx:
            current_layer = model_mp.get_layer(
                name=f'quant_{sorted_configurable_nodes_names[node_idx_to_configure]}')
            _set_layer_to_bitwidth(current_layer, mp_model_configuration[node_idx_to_configure])

    else:  # use the entire mp_model_configuration to configure the model
        for node_idx_to_configure, bitwidth_idx in enumerate(mp_model_configuration):
            current_layer = model_mp.get_layer(
                name=f'quant_{sorted_configurable_nodes_names[node_idx_to_configure]}')
            _set_layer_to_bitwidth(current_layer, mp_model_configuration[node_idx_to_configure])


def _build_distance_matrix(baseline_tensors: List[Tensor],
                           mp_tensors: List[Tensor],
                           compute_distance_fn: Callable):
    """
    Compute the distance between the MP model's outputs and the baseline model's outputs
    for each image in the batch that was inferred.

    Args:
        baseline_tensors: Baseline model's output tensors.
        mp_tensors: MP model's output tensors.
        compute_distance_fn: Function to compute the distance between two tensors.

    Returns:
        A distance matrix that maps each node's index to the distance between this node's output
         and the baseline model's output for all images that were inferred.
    """

    num_interest_points = len(baseline_tensors)
    num_samples = len(baseline_tensors[0])
    distance_matrix = np.ndarray((num_interest_points, num_samples))
    for i in range(num_interest_points):
        for j in range(num_samples):
            distance_matrix[i, j] = compute_distance_fn(baseline_tensors[i][j].numpy(), mp_tensors[i][j].numpy())
    return distance_matrix


def _build_baseline_model(graph: Graph,
                          interest_points: List[BaseNode]) -> Model:
    """
    Build a Keras baseline model to compare inferences of the MP model to.
    The baseline model is the float model we build from the graph.

    Args:
        graph: Graph to build its baseline Keras model.
        interest_points: List of nodes to get their outputs.

    Returns:
        A baseline Keras model.
    """

    baseline_model, _ = model_builder(graph,
                                      mode=ModelBuilderMode.FLOAT,
                                      append2output=interest_points)
    return baseline_model


def _tensors_as_list(tensors: Any) -> List[Any]:
    """
    Create a list of tensors if they are not in a list already.
    This functions helpful when the graph has only one node (so the model's output is a Tensor and not a list of
    Tensors).

    Args:
        tensors: Tensors to return as a list.

    Returns:
        List of tensors.
    """
    if not isinstance(tensors, list):
        return [tensors]
    return tensors


def _set_layer_to_bitwidth(wrapped_layer: Layer,
                           bitwidth_idx: int):
    """
    Configure a layer (which is wrapped in a QuantizeWrapper and holds a
    SelectiveWeightsQuantizeConfig in its quantize_config) to work with a different bitwidth.
    The bitwidth_idx is the index of the quantized-weights the quantizer in the SelectiveWeightsQuantizeConfig holds.

    Args:
        wrapped_layer: Layer to change its bitwidth.
        bitwidth_idx: Index of the bitwidth the layer should work with.

    """
    assert isinstance(wrapped_layer, QuantizeWrapper) and isinstance(wrapped_layer.quantize_config,
                                                                     SelectiveWeightsQuantizeConfig)
    # Configure the quantize_config to use a different bitwidth
    # (in practice, to use a different already quantized kernel).
    wrapped_layer.quantize_config.set_bit_width_index(bitwidth_idx)
