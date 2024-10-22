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
from typing import Callable, Any, List, Tuple

from model_compression_toolkit.constants import AXIS
from model_compression_toolkit.core import FrameworkInfo, MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.similarity_analyzer import compute_kl_divergence
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.hessian import HessianScoresRequest, HessianMode, \
    HessianScoresGranularity, HessianInfoService


class SensitivityEvaluation:
    """
    Class to wrap and manage the computation on distance metric for Mixed-Precision quantization search.
    It provides a function that evaluates the sensitivity of a bit-width configuration for the MP model.
    """

    def __init__(self,
                 graph: Graph,
                 quant_config: MixedPrecisionQuantizationConfig,
                 representative_data_gen: Callable,
                 fw_info: FrameworkInfo,
                 fw_impl: Any,
                 set_layer_to_bitwidth: Callable,
                 disable_activation_for_metric: bool = False,
                 hessian_info_service: HessianInfoService = None
                 ):
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
            disable_activation_for_metric: Whether to disable activation quantization when computing the MP metric.
            hessian_info_service: HessianInfoService to fetch Hessian approximation information.

        """
        self.graph = graph
        self.quant_config = quant_config
        self.representative_data_gen = representative_data_gen
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.set_layer_to_bitwidth = set_layer_to_bitwidth
        self.disable_activation_for_metric = disable_activation_for_metric
        if self.quant_config.use_hessian_based_scores:
            if not isinstance(hessian_info_service, HessianInfoService):
                Logger.critical(
                    f"When using Hessian-based approximations for sensitivity evaluation, a valid HessianInfoService object is required; found {type(hessian_info_service)}.")
            self.hessian_info_service = hessian_info_service

        self.sorted_configurable_nodes_names = graph.get_configurable_sorted_nodes_names(self.fw_info)

        # Get interest points and output points set for distance measurement and set other helper datasets
        # We define a separate set of output nodes of the model for the purpose of sensitivity computation.
        self.interest_points = get_mp_interest_points(graph,
                                                      fw_impl.count_node_for_mixed_precision_interest_points,
                                                      quant_config.num_interest_points_factor)

        # We use normalized MSE when not running hessian-based. For Hessian-based normalized MSE is not needed
        # beacause hessian weights already do normalization.
        use_normalized_mse = self.quant_config.use_hessian_based_scores is False
        self.ips_distance_fns, self.ips_axis = self._init_metric_points_lists(self.interest_points, use_normalized_mse)

        self.output_points = get_output_nodes_for_metric(graph)
        self.out_ps_distance_fns, self.out_ps_axis = self._init_metric_points_lists(self.output_points,
                                                                                    use_normalized_mse)

        # Setting lists with relative position of the interest points
        # and output points in the list of all mp model activation tensors
        graph_sorted_nodes = self.graph.get_topo_sorted_nodes()
        all_out_tensors_indices = [graph_sorted_nodes.index(n) for n in self.interest_points + self.output_points]
        global_ipts_indices = [graph_sorted_nodes.index(n) for n in self.interest_points]
        global_out_pts_indices = [graph_sorted_nodes.index(n) for n in self.output_points]
        self.ips_act_indices = [all_out_tensors_indices.index(i) for i in global_ipts_indices]
        self.out_ps_act_indices = [all_out_tensors_indices.index(i) for i in global_out_pts_indices]

        # Build a mixed-precision model which can be configured to use different bitwidth in different layers.
        # And a baseline model.
        # Also, returns a mapping between a configurable graph's node and its matching layer(s)
        # in the new built MP model.
        self.baseline_model, self.model_mp, self.conf_node2layers = self._build_models()

        # Build images batches for inference comparison
        self.images_batches = self._get_images_batches(quant_config.num_of_images)

        # Get baseline model inference on all samples
        self.baseline_tensors_list = []  # setting from outside scope

        # Casting images tensors to the framework tensor type.
        self.images_batches = list(map(lambda in_arr: self.fw_impl.to_tensor(in_arr), self.images_batches))

        # Initiating baseline_tensors_list since it is not initiated in SensitivityEvaluationManager init.
        self._init_baseline_tensors_list()

        # Computing Hessian-based scores for weighted average distance metric computation (only if requested),
        # and assigning distance_weighting method accordingly.
        self.interest_points_hessians = None
        if self.quant_config.use_hessian_based_scores is True:
            self.interest_points_hessians = self._compute_hessian_based_scores()
            self.quant_config.distance_weighting_method = lambda d: self.interest_points_hessians

    def _init_metric_points_lists(self, points: List[BaseNode], norm_mse: bool = False) -> Tuple[
        List[Callable], List[int]]:
        """
        Initiates required lists for future use when computing the sensitivity metric.
        Each point on which the metric is computed uses a dedicated distance function based on its type.
        In addition, all distance functions preform batch computation. Axis is needed only for KL Divergence computation.

        Args:
            points: The set of nodes in the graph for which we need to initiate the lists.
            norm_mse: whether to normalize mse distance function.

        Returns: A lists with distance functions and an axis list for each node.

        """
        distance_fns_list = []
        axis_list = []
        for n in points:
            distance_fn, axis = self.fw_impl.get_mp_node_distance_fn(n,
                                                                     compute_distance_fn=self.quant_config.compute_distance_fn,
                                                                     norm_mse=norm_mse)
            distance_fns_list.append(distance_fn)
            # Axis is needed only for KL Divergence calculation, otherwise we use per-tensor computation
            axis_list.append(axis if distance_fn == compute_kl_divergence else None)
        return distance_fns_list, axis_list

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
        self._configure_bitwidths_model(mp_model_configuration,
                                        node_idx)

        # Compute the distance metric
        ipts_distances, out_pts_distances = self._compute_distance()

        # Configure MP model back to the same configuration as the baseline model if baseline provided
        if baseline_mp_configuration is not None:
            self._configure_bitwidths_model(baseline_mp_configuration,
                                            node_idx)

        return self._compute_mp_distance_measure(ipts_distances, out_pts_distances,
                                                 self.quant_config.distance_weighting_method)

    def _init_baseline_tensors_list(self):
        """
        Evaluates the baseline model on all images and saves the obtained lists of tensors in a list for later use.
        Initiates a class variable self.baseline_tensors_list
        """
        self.baseline_tensors_list = [self.fw_impl.to_numpy(self.fw_impl.sensitivity_eval_inference(self.baseline_model,
                                                                                                    images))
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

        model_mp, _, conf_node2layers = self.fw_impl.model_builder(evaluation_graph,
                                                                   mode=ModelBuilderMode.MIXEDPRECISION,
                                                                   append2output=self.interest_points + self.output_points,
                                                                   fw_info=self.fw_info)

        # Build a baseline model.
        baseline_model, _ = self.fw_impl.model_builder(evaluation_graph,
                                                       mode=ModelBuilderMode.FLOAT,
                                                       append2output=self.interest_points + self.output_points)

        return baseline_model, model_mp, conf_node2layers

    def _compute_hessian_based_scores(self) -> np.ndarray:
        """
        Compute Hessian-based scores for each interest point.

        Returns: A vector of scores, one for each interest point,
         to be used for the distance metric weighted average computation.

        """
        # Create a request for Hessian approximation scores with specific configurations
        # (here we use per-tensor approximation of the Hessian's trace w.r.t the node's activations)
        fw_dataloader = self.fw_impl.convert_data_gen_to_dataloader(self.representative_data_gen,
                                                                    batch_size=self.quant_config.hessian_batch_size)
        hessian_info_request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                                    granularity=HessianScoresGranularity.PER_TENSOR,
                                                    target_nodes=self.interest_points,
                                                    data_loader=fw_dataloader,
                                                    n_samples=self.quant_config.num_of_images)

        # Fetch the Hessian approximation scores for the current interest point
        nodes_approximations = self.hessian_info_service.fetch_hessian(request=hessian_info_request)
        approx_by_image = np.stack([nodes_approximations[n.name] for n in self.interest_points], axis=1)    # samples X nodes

        # Return the mean approximation value across all images for each interest point
        return np.mean(approx_by_image, axis=0)

    def _configure_bitwidths_model(self,
                                   mp_model_configuration: List[int],
                                   node_idx: List[int]):
        """
        Configure a dynamic model (namely, model with layers that their weights and activation
        bit-width can be configured) using an MP model configuration mp_model_configuration.

        Args:
            mp_model_configuration: Configuration of bit-width indices to set to the model.
            node_idx: List of nodes' indices to configure (the rest layers are configured as the baseline model).
        """

        # Configure model
        # Note: Not all nodes in the graph are included in the MP model that is returned by the model builder.
        # Thus, the last configurable layer must be included in the interest points for evaluating the metric,
        # otherwise, not all configurable nodes will be considered throughout the MP optimization search (since
        # they will not affect the metric value).
        if node_idx is not None:  # configure specific layers in the mp model
            for node_idx_to_configure in node_idx:
                self._configure_node_bitwidth(self.sorted_configurable_nodes_names,
                                              mp_model_configuration, node_idx_to_configure)
        else:  # use the entire mp_model_configuration to configure the model
            for node_idx_to_configure, bitwidth_idx in enumerate(mp_model_configuration):
                self._configure_node_bitwidth(self.sorted_configurable_nodes_names,
                                              mp_model_configuration, node_idx_to_configure)

    def _configure_node_bitwidth(self,
                                 sorted_configurable_nodes_names: List[str],
                                 mp_model_configuration: List[int],
                                 node_idx_to_configure: int):
        """
        Configures a node with multiple quantization candidates to the bitwidth candidate in the given index.
        Args:
            sorted_configurable_nodes_names: A list of configurable nodes names sorted according to the graph
                topological sort order.
            mp_model_configuration: Configuration of bit-width indices to set to the model.
            node_idx_to_configure: Quantization configuration candidate to configure.

        Returns:

        """
        node_name = sorted_configurable_nodes_names[node_idx_to_configure]
        layers_to_config = self.conf_node2layers.get(node_name, None)
        if layers_to_config is None:
            Logger.critical(
                f"Matching layers for node {node_name} not found in the mixed precision model configuration.")  # pragma: no cover

        for current_layer in layers_to_config:
            self.set_layer_to_bitwidth(current_layer, mp_model_configuration[node_idx_to_configure])

    def _compute_points_distance(self,
                                 baseline_tensors: List[Any],
                                 mp_tensors: List[Any],
                                 points_distance_fns: List[Callable],
                                 points_axis: List[int]):
        """
        Compute the distance on the given set of points outputs between the MP model and the baseline model
        for each image in the batch that was inferred.

        Args:
            baseline_tensors: Baseline model's output tensors of the given points.
            mp_tensors: MP model's output tensors pf the given points.
            points_distance_fns: A list with distance function to compute the distance between each given
                point's output tensors.
            points_axis: A list with the matching axis of each given point's output tensors.

        Returns:
            A distance vector that maps each node's index in the given nodes list to the distance between this node's output
             and the baseline model's output for all images that were inferred.
        """

        distance_v = [fn(x, y, batch=True, axis=axis) for fn, x, y, axis
                      in zip(points_distance_fns, baseline_tensors, mp_tensors, points_axis)]

        return np.asarray(distance_v)

    def _compute_distance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computing the interest points distance and the output points distance, and using them to build a
        unified distance vector.

        Returns: A distance vector.
        """

        ipts_per_batch_distance = []
        out_pts_per_batch_distance = []

        # Compute the distance matrix for num_of_images images.
        for images, baseline_tensors in zip(self.images_batches, self.baseline_tensors_list):
            # when using model.predict(), it does not use the QuantizeWrapper functionality
            mp_tensors = self.fw_impl.sensitivity_eval_inference(self.model_mp, images)
            mp_tensors = self.fw_impl.to_numpy(mp_tensors)

            # Compute distance: similarity between the baseline model to the float model
            # in every interest point for every image in the batch.
            ips_distance = self._compute_points_distance([baseline_tensors[i] for i in self.ips_act_indices],
                                                         [mp_tensors[i] for i in self.ips_act_indices],
                                                         self.ips_distance_fns,
                                                         self.ips_axis)
            outputs_distance = self._compute_points_distance([baseline_tensors[i] for i in self.out_ps_act_indices],
                                                             [mp_tensors[i] for i in self.out_ps_act_indices],
                                                             self.out_ps_distance_fns,
                                                             self.out_ps_axis)

            # Extending the dimensions for the concatenation at the end in case we need to
            ips_distance = ips_distance if len(ips_distance.shape) > 1 else ips_distance[:, None]
            outputs_distance = outputs_distance if len(outputs_distance.shape) > 1 else outputs_distance[:, None]
            ipts_per_batch_distance.append(ips_distance)
            out_pts_per_batch_distance.append(outputs_distance)

        # Merge all distance matrices into a single distance matrix.
        ipts_distances = np.concatenate(ipts_per_batch_distance, axis=1)
        out_pts_distances = np.concatenate(out_pts_per_batch_distance, axis=1)

        return ipts_distances, out_pts_distances

    @staticmethod
    def _compute_mp_distance_measure(ipts_distances: np.ndarray,
                                     out_pts_distances: np.ndarray,
                                     metrics_weights_fn: Callable) -> float:
        """
        Computes the final distance value out of a distance matrix.

        Args:
            ipts_distances: A matrix that contains the distances between the baseline and MP models
                for each interest point.
            out_pts_distances: A matrix that contains the distances between the baseline and MP models
                for each output point.
            metrics_weights_fn: A callable that produces the scores to compute weighted distance for interest points.

        Returns: Distance value.
        """
        mean_ipts_distance = 0
        if len(ipts_distances) > 0:
            mean_distance_per_layer = ipts_distances.mean(axis=1)

            # Use weights such that every layer's distance is weighted differently (possibly).
            weight_scores = metrics_weights_fn(ipts_distances)
            weight_scores = np.asarray(weight_scores) if isinstance(weight_scores, List) else weight_scores
            weight_scores = weight_scores.flatten()

            mean_ipts_distance = np.average(mean_distance_per_layer, weights=weight_scores)

        mean_output_distance = 0
        if len(out_pts_distances) > 0:
            mean_distance_per_output = out_pts_distances.mean(axis=1)
            mean_output_distance = np.average(mean_distance_per_output)

        return mean_output_distance + mean_ipts_distance

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
        for inference_batch_input in self.representative_data_gen():
            if samples_count >= num_of_images:
                break
            batch_size = inference_batch_input[0].shape[0]

            # If we sampled more images than we should use in the distance matrix,
            # we take only a subset of these images and use only them for computing the distance matrix.
            if batch_size > num_of_images - samples_count:
                inference_batch_input = [x[:num_of_images - samples_count] for x in inference_batch_input]
                assert num_of_images - samples_count == inference_batch_input[0].shape[0]
                batch_size = num_of_images - samples_count

            images_batches.append(inference_batch_input)
            samples_count += batch_size
        else:
            if samples_count < num_of_images:
                Logger.warning(f'Not enough images in representative dataset to generate {num_of_images} data points, '
                               f'only {samples_count} were generated')
        return images_batches


def get_mp_interest_points(graph: Graph,
                           interest_points_classifier: Callable,
                           num_ip_factor: float) -> List[BaseNode]:
    """
    Gets a list of interest points for the mixed precision metric computation.
    The list is constructed from a filtered set of nodes in the graph.
    Note that the output layers are separated from the interest point set for metric computation purposes.

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

    # We exclude output nodes from the set of interest points since they are used separately in the sensitivity evaluation.
    output_nodes = [n.node for n in graph.get_outputs()]

    interest_points = [n for n in interest_points_nodes if n not in output_nodes]

    return interest_points


def get_output_nodes_for_metric(graph: Graph) -> List[BaseNode]:
    """
    Returns a list of output nodes that are also quantized (either kernel weights attribute or activation)
    to be used as a set of output points in the distance metric computation.

    Args:
        graph: Graph to search for its MP configuration.

    Returns: A list of output nodes.

    """

    return [n.node for n in graph.get_outputs()
            if (graph.fw_info.is_kernel_op(n.node.type) and
                n.node.is_weights_quantization_enabled(graph.fw_info.get_kernel_op_attributes(n.node.type)[0])) or
            n.node.is_activation_quantization_enabled()]


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
