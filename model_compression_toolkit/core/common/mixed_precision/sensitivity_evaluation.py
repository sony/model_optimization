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

import numpy as np
from typing import Callable, Any, List

from model_compression_toolkit import FrameworkInfo, MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common import Logger


class SensitivityEvaluation:
    """
    Class to wrap and manage the computation on distance metric for Mixed-Precision quantization search.
    It provides a function that evaluates the sensitivity of a bit-width configuration for the MP model.
    """

    def __init__(self,
                 graph: Graph,
                 quant_config: MixedPrecisionQuantizationConfigV2,
                 representative_data_gen: Callable,
                 fw_info: FrameworkInfo,
                 fw_impl: Any,
                 set_layer_to_bitwidth: Callable,
                 get_quant_node_name: Callable,
                 disable_activation_for_metric: bool = False):
        """
        Initiates all relevant objects to manage a sensitivity evaluation for MP search.
        Create an object that allows to compute the sensitivity metric of an MP model (the sensitivity
        is computed based on the similarity of the interest points' outputs between the MP model
        and the float model).
        First, we initiate a SensitivityEvaluationManager that handles the components which are necessary for
        evaluating the sensitivity. It initializes an MP model (a model where layers that can be configured in
        different bit-widths) and a baseline model (a float model).
        Then, and based on the outputs of these two models (for some batches from the representative_data_gen),
        we build a function to measure the sensitivity of a change in a bit-width of a model's layer.

        Args:
            graph: Graph to search for its MP configuration.
            fw_info: FrameworkInfo object about the specific framework
                (e.g., attributes of different layers' weights to quantize).
            quant_config: MP Quantization configuration for how the graph should be quantized.
            representative_data_gen: Dataset used for getting batches for inference.
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            set_layer_to_bitwidth: A fw-dependent function that allows to configure a configurable MP model
                    with a specific bit-width configuration.
            get_quant_node_name: A fw-dependent function that takes a node's name and outputs the node's name in a
                quantized model (according to the fw conventions).
            disable_activation_for_metric: Whether to disable activation quantization when computing the MP metric.
        """
        self.graph = graph
        self.quant_config = quant_config
        self.representative_data_gen = representative_data_gen
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.set_layer_to_bitwidth = set_layer_to_bitwidth
        self.get_quant_node_name = get_quant_node_name
        self.disable_activation_for_metric = disable_activation_for_metric

        # Get interest points for distance measurement and a list of sorted configurable nodes names
        self.sorted_configurable_nodes_names = graph.get_configurable_sorted_nodes_names()
        self.interest_points = get_mp_interest_points(graph,
                                                      fw_impl.count_node_for_mixed_precision_interest_points,
                                                      quant_config.num_interest_points_factor)

        self.outputs_replacement_nodes = None
        self.output_nodes_indices = None
        if self.quant_config.use_grad_based_weights is True:
            # Getting output replacement (if needed) - if a model's output layer is not compatible for the task of
            # gradients computation then we find a predecessor layer which is compatible,
            # add it to the set of interest points and use it for the gradients' computation.
            # Note that we need to modify the set of interest points before building the models,
            # therefore, it is separated from the part where we compute the actual gradient weights.
            self.outputs_replacement_nodes = get_output_replacement_nodes(graph, fw_impl)
            self.output_nodes_indices = self._update_ips_with_outputs_replacements()

        # Build a mixed-precision model which can be configured to use different bitwidth in different layers.
        # And a baseline model.
        self.baseline_model, self.model_mp = self._build_models()

        # Build images batches for inference comparison
        self.images_batches = self._get_images_batches(quant_config.num_of_images)

        # Get baseline model inference on all samples
        self.baseline_tensors_list = []  # setting from outside scope

        # Casting images tensors to the framework tensor type.
        self.images_batches = list(map(lambda in_arr: self.fw_impl.to_tensor(in_arr), self.images_batches))

        # Initiating baseline_tensors_list since it is not initiated in SensitivityEvaluationManager init.
        self._init_baseline_tensors_list()

        # Computing gradient-based weights for weighted average distance metric computation (only if requested),
        # and assigning distance_weighting method accordingly.
        self.interest_points_gradients = None
        if self.quant_config.use_grad_based_weights is True:
            assert self.outputs_replacement_nodes is not None and self.output_nodes_indices is not None, \
                f"{self.outputs_replacement_nodes} and {self.output_nodes_indices} " \
                f"should've been assigned before computing the gradient-based weights."

            self.interest_points_gradients = self._compute_gradient_based_weights()
            self.quant_config.distance_weighting_method = lambda d: self.interest_points_gradients

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
        self._configure_bitwidths_model(self.model_mp,
                                        self.sorted_configurable_nodes_names,
                                        mp_model_configuration,
                                        node_idx)

        # Compute the distance matrix
        distance_matrix = self._build_distance_metrix()

        # Configure MP model back to the same configuration as the baseline model if baseline provided
        if baseline_mp_configuration is not None:
            self._configure_bitwidths_model(self.model_mp,
                                            self.sorted_configurable_nodes_names,
                                            baseline_mp_configuration,
                                            node_idx)

        return self._compute_mp_distance_measure(distance_matrix, self.quant_config.distance_weighting_method)

    def _init_baseline_tensors_list(self):
        """
        Evaluates the baseline model on all images and saves the obtained lists of tensors in a list for later use.
        Initiates a class variable self.baseline_tensors_list
        """
        self.baseline_tensors_list = [self._tensors_as_list(self.fw_impl.to_numpy(self.baseline_model(images)))
                                      for images in self.images_batches]

    def _build_models(self) -> Any:
        """
        Builds two models - an MP model with configurable layers and a baseline, float model.

        Returns: A tuple with two models built from the given graph: a baseline model (with baseline configuration) and
            an MP model (which can be configured for a specific bitwidth configuration).
            Note that the type of the returned models is dependent on the used framework (TF/Pytorch).
        """

        evaluation_graph = copy.deepcopy(self.graph)

        if self.disable_activation_for_metric:
            for n in evaluation_graph.get_topo_sorted_nodes():
                for c in n.candidates_quantization_cfg:
                    c.activation_quantization_cfg.enable_activation_quantization = False

        model_mp, _ = self.fw_impl.model_builder(evaluation_graph,
                                                 mode=ModelBuilderMode.MIXEDPRECISION,
                                                 append2output=self.interest_points,
                                                 fw_info=self.fw_info)

        # Build a baseline model.
        baseline_model, _ = self.fw_impl.model_builder(evaluation_graph,
                                                       mode=ModelBuilderMode.FLOAT,
                                                       append2output=self.interest_points)

        return baseline_model, model_mp

    def _compute_gradient_based_weights(self) -> np.ndarray:
        """
        Computes the gradient-based weights using the framework's model_grad method per batch of images.

        Returns: A vector of weights, one for each interest point,
        to be used for the distance metric weighted average computation.
        """

        grad_per_batch = []
        for images in self.images_batches:
            batch_ip_gradients = []
            for i in range(1, images[0].shape[0] + 1):
                image_ip_gradients = self.fw_impl.model_grad(self.graph,
                                                             {inode: images[0][i - 1:i] for inode in
                                                              self.graph.get_inputs()},
                                                             self.interest_points,
                                                             self.outputs_replacement_nodes,
                                                             self.output_nodes_indices,
                                                             self.quant_config.output_grad_factor,
                                                             norm_weights=self.quant_config.norm_weights)
                batch_ip_gradients.append(image_ip_gradients)
            grad_per_batch.append(np.mean(batch_ip_gradients, axis=0))
        return np.mean(grad_per_batch, axis=0)

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

    def _compute_distance_matrix(self,
                                 baseline_tensors: List[Any],
                                 mp_tensors: List[Any]):
        """
        Compute the distance between the MP model's outputs and the baseline model's outputs
        for each image in the batch that was inferred.
        Args:
            baseline_tensors: Baseline model's output tensors.
            mp_tensors: MP model's output tensors.
        Returns:
            A distance matrix that maps each node's index to the distance between this node's output
             and the baseline model's output for all images that were inferred.
        """

        assert len(baseline_tensors) == len(self.interest_points)
        num_interest_points = len(baseline_tensors)
        num_samples = len(baseline_tensors[0])
        distance_matrix = np.ndarray((num_interest_points, num_samples))

        for i in range(num_interest_points):
            point_distance_fn = \
                self.fw_impl.get_node_distance_fn(layer_class=self.interest_points[i].layer_class,
                                                  framework_attrs=self.interest_points[i].framework_attr,
                                                  compute_distance_fn=self.quant_config.compute_distance_fn)

            distance_matrix[i] = point_distance_fn(baseline_tensors[i], mp_tensors[i], batch=True)

        return distance_matrix

    def _build_distance_metrix(self):
        """
        Builds a matrix that contains the distances between the baseline and MP models for each interest point.
        Returns: A distance matrix.
        """
        # List of distance matrices. We create a distance matrix for each sample from the representative_data_gen
        # and merge all of them eventually.
        distance_matrices = []

        # Compute the distance matrix for num_of_images images.
        for images, baseline_tensors in zip(self.images_batches, self.baseline_tensors_list):
            # when using model.predict(), it does not use the QuantizeWrapper functionality
            mp_tensors = self.model_mp(images)
            mp_tensors = self._tensors_as_list(self.fw_impl.to_numpy(mp_tensors))

            # Build distance matrix: similarity between the baseline model to the float model
            # in every interest point for every image in the batch.
            distance_matrices.append(self._compute_distance_matrix(baseline_tensors, mp_tensors))

        # Merge all distance matrices into a single distance matrix.
        distance_matrix = np.concatenate(distance_matrices, axis=1)

        # Assert we used a correct number of images for computing the distance matrix
        assert distance_matrix.shape[1] == self.quant_config.num_of_images
        return distance_matrix

    @staticmethod
    def _compute_mp_distance_measure(distance_matrix: np.ndarray, metrics_weights_fn: Callable) -> float:
        """
        Computes the final distance value out of a distance matrix.

        Args:
            distance_matrix: A matrix that contains the distances between the baseline and MP models
                for each interest point.
            metrics_weights_fn:

        Returns: Distance value.
        """
        # Compute the distance between the baseline model's outputs and the MP model's outputs.
        # The distance is the mean of distances over all images in the batch that was inferred.
        mean_distance_per_layer = distance_matrix.mean(axis=1)
        # Use weights such that every layer's distance is weighted differently (possibly).
        return np.average(mean_distance_per_layer, weights=metrics_weights_fn(distance_matrix))

    def _get_images_batches(self, num_of_images: int) -> List[Any]:
        """
        Construct batches of image samples for inference.

        Args:
            num_of_images: Num of total images for evaluation.

        Returns: A list of images batches (lists of images)
        """
        # First, select images to use for all measurements.
        samples_count = 0  # Number of images we used so far to compute the distance matrix.
        images_batches = []
        while samples_count < num_of_images:
            # Get a batch of images to infer in both models.
            inference_batch_input = self.representative_data_gen()
            batch_size = inference_batch_input[0].shape[0]

            # If we sampled more images than we should use in the distance matrix,
            # we take only a subset of these images and use only them for computing the distance matrix.
            if batch_size > num_of_images - samples_count:
                inference_batch_input = [x[:num_of_images - samples_count] for x in inference_batch_input]
                assert num_of_images - samples_count == inference_batch_input[0].shape[0]
                batch_size = num_of_images - samples_count

            images_batches.append(inference_batch_input)
            samples_count += batch_size
        return images_batches

    def _update_ips_with_outputs_replacements(self):
        """
        Updates the list of interest points with the set of pre-calculated replacement outputs.
        Also, returns the indices of all output nodes (original, replacements and nodes in between them) in a
        topological sorted interest points list (for later use in gradients computation and normalization).

        Returns: A list of indices of the output nodes in the sorted interest points list.

        """

        assert self.outputs_replacement_nodes is not None, \
            "Trying to update interest points list with new output nodes but outputs_replacement_nodes list is None."

        replacement_outputs_to_ip = [r_node for r_node in self.outputs_replacement_nodes if
                                     r_node not in self.interest_points]
        updated_interest_points = self.interest_points + replacement_outputs_to_ip

        # Re-sort interest points in a topological order according to the graph's sort
        self.interest_points = [n for n in self.graph.get_topo_sorted_nodes() if n in updated_interest_points]

        output_indices = [self.interest_points.index(n.node) for n in self.graph.get_outputs()]
        replacement_indices = [self.interest_points.index(n) for n in self.outputs_replacement_nodes]
        return list(set(output_indices + replacement_indices))

    @staticmethod
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


def get_mp_interest_points(graph: Graph,
                           interest_points_classifier: Callable,
                           num_ip_factor: float) -> List[BaseNode]:
    """
    Gets a list of interest points for the mixed precision metric computation.
    The list is constructed from a filtered set of the convolutions nodes in the graph.

    Args:
        graph: Graph to search for its MP configuration.
        interest_points_classifier: A function that indicates whether a given node in considered as a potential
            interest point for mp metric computation purposes.
        num_ip_factor: Percentage out of the total set of interest points that we want to actually use.

    Returns: A list of interest points (nodes in the graph).

    """
    sorted_nodes = graph.get_topo_sorted_nodes()
    ip_nodes = list(filter(lambda n: interest_points_classifier(n), sorted_nodes))

    interest_points_nodes = bound_num_interest_points(ip_nodes, num_ip_factor)

    # We add output layers of the model to interest points
    # in order to consider the model's output in the distance metric computation (and also to make sure
    # all configurable layers are included in the configured mp model for metric computation purposes)
    output_nodes = [n.node for n in graph.get_outputs() if n.node not in interest_points_nodes]
    interest_points = interest_points_nodes + output_nodes

    return interest_points


def get_output_replacement_nodes(graph: Graph,
                                 fw_impl: Any) -> List[BaseNode]:
    """
    If a model's output node is not compatible for the task of gradients computation we need to find a predecessor
    node in the model's graph representation which is compatible and add it to the set of interest points and use it
    for the gradients' computation. This method searches for this predecessor node for each output of the model.

    Args:
        graph: Graph to search for replacement output nodes.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns: A list of output replacement nodes.

    """
    replacement_outputs = []
    for n in graph.get_outputs():
        prev_node = n.node
        while not fw_impl.is_node_compatible_for_metric_outputs(prev_node):
            prev_node = graph.get_prev_nodes(n.node)
            assert len(prev_node) == 1, "A none MP compatible output node has multiple inputs, " \
                                        "which is incompatible for metric computation."
            prev_node = prev_node[0]
        replacement_outputs.append(prev_node)
    return replacement_outputs


def bound_num_interest_points(sorted_ip_list: List[BaseNode], num_ip_factor: float) -> List[BaseNode]:
    """
    Filters the list of interest points and returns a shorter list with number of interest points smaller than some
    default threshold.

    Args:
        sorted_ip_list: List of nodes which are considered as interest points for the metric computation.
        num_ip_factor: Percentage out of the total set of interest points that we want to actually use.

    Returns: A new list of interest points (list of nodes).

    """
    if num_ip_factor < 1.0:
        num_interest_points = int(num_ip_factor * len(sorted_ip_list))
        Logger.info(f'Using {num_interest_points} for mixed-precision metric evaluation out of total '
                    f'{len(sorted_ip_list)} potential interest points.')
        # Take num_interest_points evenly spaced interest points from the original list
        indices = np.round(np.linspace(0, len(sorted_ip_list) - 1, num_interest_points)).astype(int)
        return [sorted_ip_list[i] for i in indices]

    return sorted_ip_list
