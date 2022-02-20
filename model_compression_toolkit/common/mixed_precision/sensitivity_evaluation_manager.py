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
import numpy as np
from typing import Callable, Any, List

from model_compression_toolkit import FrameworkInfo, MixedPrecisionQuantizationConfig
from model_compression_toolkit.common import Graph, BaseNode
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode


class SensitivityEvaluationManager:
    """
    Class to wrap and manage the computation on distance metric for Mixed-Precision quantization search.
    """

    def __init__(self,
                 graph: Graph,
                 fw_info: FrameworkInfo,
                 quant_config: MixedPrecisionQuantizationConfig,
                 representative_data_gen: Callable,
                 model_builder: Callable,
                 move_tensors_func: Callable = None):
        """
        Initiates all relevant objects to manage a sensitivity evaluation for MP search.
        Args:
            graph: Graph to search for its MP configuration.
            fw_info: FrameworkInfo object about the specific framework
                (e.g., attributes of different layers' weights to quantize).
            quant_config: MP Quantization configuration for how the graph should be quantized.
            representative_data_gen: Dataset used for getting batches for inference.
            model_builder: A function that builds a model object for the currently used framework.
            move_tensors_func: A function that moves tensors in list of tensors to cpu memory
                (relevant only when using Pytorch framework).
        """
        self.quant_config = quant_config

        # If used with Pytorch, we might need to move tensors to cpu before applying numpy functions
        self.move_tensors_func = move_tensors_func

        # Get interest points for distance measurement and a list of sorted configurable nodes names
        self.sorted_configurable_nodes_names = graph.get_configurable_sorted_nodes_names()
        self.interest_points = graph.get_configurable_sorted_nodes()

        # Build a mixed-precision model which can be configured to use different bitwidth in different layers.
        # And a baseline model.
        self.baseline_model, self.model_mp = build_models(graph, fw_info, self.interest_points, model_builder)

        # Build images batches for inference comparison
        self.images_batches = get_images_batches(quant_config.num_of_images, representative_data_gen)

        # Get baseline model inference on all samples
        self.baseline_tensors_list = []  # setting from outside scope

    def init_baseline_tensors_list(self):
        """
        Evaluates the baseline model on all images and saves the obtained lists of tensors in a list for later use.
        Initiates a class variable self.baseline_tensors_list
        """
        self._get_baseline_tensors()
        if self.move_tensors_func:
            # for Pytorch framework use only
            self.baseline_tensors_list = [self.move_tensors_func(tensors) for tensors in self.baseline_tensors_list]

    def _get_baseline_tensors(self):
        """
        Evaluate the baseline model on each of the images batches and constructs a list with the results for each batch.
        Sets a class variable with the results.
        """
        # Infer all images with baseline model, and save them as a list.
        # If the model contains only one output we save it a list. If it's a list already, we keep it as a list.
        self.baseline_tensors_list = [_tensors_as_list(self.baseline_model(images)) for images in self.images_batches]

    def compute_distance_matrix(self,
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

        num_interest_points = len(baseline_tensors)
        num_samples = len(baseline_tensors[0])
        distance_matrix = np.ndarray((num_interest_points, num_samples))
        for i in range(num_interest_points):
            for j in range(num_samples):
                distance_matrix[i, j] = \
                    self.quant_config.compute_distance_fn(baseline_tensors[i][j].numpy(), mp_tensors[i][j].numpy())
        return distance_matrix

    def build_distance_metrix(self):
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
            mp_tensors = _tensors_as_list(self.model_mp(images))

            if self.move_tensors_func:
                # in case the tensors are on GPU, and we use Pytorch,
                # need to move them to CPU in order to convert to numpy arrays
                mp_tensors = self.move_tensors_func(mp_tensors)

            # Build distance matrix: similarity between the baseline model to the float model
            # in every interest point for every image in the batch.
            distance_matrices.append(self.compute_distance_matrix(baseline_tensors, mp_tensors))

        # Merge all distance matrices into a single distance matrix.
        distance_matrix = np.concatenate(distance_matrices, axis=1)

        # Assert we used a correct number of images for computing the distance matrix
        assert distance_matrix.shape[1] == self.quant_config.num_of_images
        return distance_matrix


def build_models(graph: Graph, fw_info: FrameworkInfo, interest_points: List[BaseNode], model_builder: Callable) -> Any:
    """
    Args:
        graph: Graph to search for its MP configuration.
        fw_info: FrameworkInfo object about the specific framework
                (e.g., attributes of different layers' weights to quantize).
        interest_points: A sorted list of graph nodes that constitute as points to compute distance measurement.
        model_builder:
    Returns: A tuple with two models built from the given graph: a baseline model (with baseline configuration) and
        an MP model (which can be configured for a specific bitwidth configuration).
        Note that the type of the returned models is dependent on the used framework (TF/Pytorch).
    """
    # Build a mixed-precision model which can be configured to use different bitwidth in different layers.
    model_mp, _ = model_builder(graph,
                                mode=ModelBuilderMode.MIXEDPRECISION,
                                append2output=interest_points,
                                fw_info=fw_info)

    # Build a baseline model.
    baseline_model, _ = model_builder(graph,
                                      mode=ModelBuilderMode.FLOAT,
                                      append2output=interest_points)

    return baseline_model, model_mp,


def get_images_batches(num_of_images: int, representative_data_gen: Callable) -> List[Any]:
    """
    Construct batches of image samples for inference.
    Args:
        num_of_images: Num of total images for evaluation.
        representative_data_gen: ataset used for getting batches for inference.
    Returns: A list of images batches (lists of images)
    """
    # First, select images to use for all measurements.
    samples_count = 0  # Number of images we used so far to compute the distance matrix.
    images_batches = []
    while samples_count < num_of_images:
        # Get a batch of images to infer in both models.
        inference_batch_input = representative_data_gen()
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


def compute_mp_distance_measure(distance_matrix, metrics_weights_fn):
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