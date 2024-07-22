# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from abc import abstractmethod
from typing import Callable, Any, Dict, Tuple, List

import numpy as np

from model_compression_toolkit.data_generation.common.enums import ImageGranularity
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline
from model_compression_toolkit.data_generation.common.model_info_exctractors import ActivationExtractor, \
    OriginalBNStatsHolder


class ImagesOptimizationHandler:
    """
    Handles the optimization process for generating images. Manages the order for which
    the image batches are optimized per iteration.

    Methods for a specific batch (specified by batch index):
    - Clear gradients.
    - Compute the batch-norm loss.
    - Perform a single optimization step.
    - Updates batch statistics.

    """

    def __init__(self,
                 model: Any,
                 data_gen_batch_size: int,
                 init_dataset: Any,
                 optimizer: Any,
                 image_pipeline: BaseImagePipeline,
                 activation_extractor: ActivationExtractor,
                 image_granularity: ImageGranularity,
                 scheduler_step_fn: Callable,
                 scheduler: Any,
                 normalization_mean: List[float],
                 normalization_std: List[float],
                 initial_lr: float,
                 eps: float = 1e-6):
        """
        Constructor for the ImagesOptimizationHandler class.

        Args:
            model (Any): The framework model used for optimization.
            data_gen_batch_size (int): The batch size for data generation.
            init_dataset (Any): The initial dataset used for images generation.
            optimizer (Any): The optimizer responsible for updating model parameters during optimization.
            image_pipeline (BaseImagePipeline): The image pipeline for processing images during optimization.
            activation_extractor (ActivationExtractor): Extractor for layer activations.
            image_granularity (ImageGranularity): The granularity level for image dependence when generating images.
            scheduler_step_fn (Callable): A function that defines the scheduler step behavior.
            scheduler (Any): The scheduler responsible for adjusting the learning rate of the optimizer over time.
            normalization_mean (List[float]): Mean values used for image normalization.
            normalization_std (List[float]): Standard deviation values used for image normalization.
            initial_lr (float): The initial learning rate used by the optimizer.
            eps (float, optional): A small value added for numerical stability. Defaults to 1e-6.

        """
        self.model = model
        self.image_pipeline = image_pipeline
        self.batch_size = data_gen_batch_size
        self.scheduler = scheduler
        self.scheduler_step_fn = scheduler_step_fn
        self.image_granularity = image_granularity
        self.eps = eps
        self.targets = []
        self.initial_lr = initial_lr
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.use_all_data_stats = False

        # Determine if all data statistics should be used
        self.use_all_data_stats = image_granularity == ImageGranularity.AllImages

    def random_batch_reorder(self):
        """
        Randomly reorders the batch indices.
        This is done to ensure that the optimization process doesn't repeatedly
        target the same batch of images at the same order.
        """
        self.rand_batch_inds = np.random.choice(self.n_batches, self.n_batches, replace=False)

    def get_random_batch_index(self, index: int) -> int:
        """
        Get the random batch index at the specified index.
        Providing a reference index that the method will use as a base to
        calculate the random batch index.

         Args:
            index (int): The index.

        Returns:
            int: The random batch index.
        """
        return self.rand_batch_inds[index]

    def get_images_by_batch_index(self, batch_index: int) -> Any:
        """
        Retrieves the images from the batch optimization holder at the given batch index.

        Args:
            batch_index (int): The index of the batch optimization holder.

        Returns:
            Any: The images in the batch.
        """
        return self.batch_opt_holders_list[batch_index].get_images()

    def get_optimizer_by_batch_index(self, batch_index: int) -> Any:
        """
        Get the optimizer for the specific batch specified by the batch index.

        Args:
            batch_index (int) : the index of the batch.

        Returns:
            Any: the optimizer of the specific batch specified by the batch index.
        """
        return self.batch_opt_holders_list[batch_index].get_optimizer()

    def get_scheduler_by_batch_index(self, batch_index: int) -> Any:
        """
        Get the scheduler for the specific batch specified by the batch index.

        Args:
            batch_index (str): the index of the batch.

        Returns:
            Any: the scheduler of the specific batch specified by the batch index.
        """
        return self.batch_opt_holders_list[batch_index].get_scheduler()

    def get_layer_accumulated_stats(self, layer_name: str) -> Tuple[Any, Any]:
        """
        Get the accumulated activation statistics for a layer.

        Args:
            layer_name (str): the name of the layer.

        Returns:
            Tuple[Any, Any]: the averaged activation statistics on all the batches for the specified layer.
        """
        raise NotImplemented   # pragma: no cover

    def compute_bn_loss(self,
                        input_imgs: Any,
                        batch_index: int,
                        activation_extractor: ActivationExtractor,
                        orig_bn_stats_holder: OriginalBNStatsHolder,
                        bn_alignment_loss_fn: Callable,
                        bn_layer_weights: Dict) -> Any:
        """
        Compute the batch norm alignment loss.

        Args:
            input_imgs (Any): the input images.
            batch_index (int): the index of the batch.
            activation_extractor (ActivationExtractor): extractor for layer activations.
            orig_bn_stats_holder (OriginalBNStatsHolder): holder for original BatchNorm statistics.
            bn_alignment_loss_fn (Callable): the batch norm alignment loss function.
            bn_layer_weights (Dict): weights to multiply the loss for each layer.

        Returns:
            Any: the computed batch norm alignment loss.
        """
        # Update the batch statistics for the current batch
        self.all_imgs_stats_holder.update_batch_stats(batch_index=batch_index,
                                                      input_imgs=input_imgs,
                                                      activation_extractor=activation_extractor,
                                                      to_differentiate=True)

        # Initialize variables for accumulating the batchnorm alignment loss
        total_bn_loss = 0

        # Iterate over each BN layer
        for layer_name in orig_bn_stats_holder.get_bn_layer_names():
            # Get the layer weight for the current BN layer
            bn_layer_weight = bn_layer_weights.get(layer_name)

            # Get the mean and variance from the original BN statistics
            bn_layer_mean = orig_bn_stats_holder.get_mean(layer_name)
            bn_layer_std = orig_bn_stats_holder.get_std(layer_name)

            # Get the mean and variance from the current batch's statistics
            if self.use_all_data_stats:
                # If using all data statistics, retrieve the accumulated statistics from all batches
                imgs_layer_mean, imgs_layer_std = self.get_layer_accumulated_stats(layer_name)
            else:
                # Otherwise, retrieve the statistics from the current batch
                imgs_layer_mean, imgs_layer_second_moment, imgs_layer_std = self.all_imgs_stats_holder.get_stats(
                    batch_index, layer_name)

            if imgs_layer_mean is not None and imgs_layer_std is not None:
                bn_alignment_loss = bn_alignment_loss_fn(bn_layer_mean, imgs_layer_mean, bn_layer_std,
                                     imgs_layer_std)
                # Accumulate the batchnorm alignment weighted by the layer weight
                total_bn_loss += bn_layer_weight * bn_alignment_loss

        return total_bn_loss

    def update_statistics(self,
                          input_imgs: Any,
                          batch_index: int,
                          activation_extractor: ActivationExtractor):
        """
        Update the statistics for the images at the specified batch index.

        Args:
            input_imgs (Any): the input images.
            batch_index (int): the index of the batch.
            activation_extractor (ActivationExtractor): extractor for layer activations.
        """
        self.all_imgs_stats_holder.update_batch_stats(batch_index=batch_index,
                                                      input_imgs=input_imgs,
                                                      activation_extractor=activation_extractor,
                                                      to_differentiate=False)

    @abstractmethod
    def optimization_step(self,
                          batch_index: int,
                          loss: Any,
                          i_ter: int):
        """
        Perform an optimization step.

        Args:
            batch_index (int): Index of the batch.
            loss (Any): The loss value.
            i_ter (int): The current iteration.
        """
        raise NotImplemented   # pragma: no cover

    @abstractmethod
    def zero_grad(self, batch_index: int):
        """
        Zero the gradients of the optimizer for the specified batch.

        Args:
            batch_index (int): Index of the batch.
        """
        raise NotImplemented   # pragma: no cover

    @abstractmethod
    def get_finalized_images(self) -> list:
        """
        Create and return a list of the generated images.

        Returns:
            list: a list of the generated images.
        """
        raise NotImplemented   # pragma: no cover


class BatchOptimizationHolder:
    """
    Holds optimization parameters for a batch of images.

    This class acts as a container for optimization-related parameters specific to a batch of images. It does not
    directly manage or handle the optimization process itself but rather holds the necessary components for
    optimization, including images, optimizer and scheduler.
    """

    def __init__(self,
                 images: Any,
                 optimizer: Any,
                 scheduler: Any,
                 initial_lr: float):
        """
        Constructor for the BatchOptimizationHolder class.

        Args:
            images (Any): a tensor containing the input images.
            optimizer (Any): optimizer responsible for updating the image parameters during optimization.
            scheduler (Any): scheduler responsible for adjusting the learning rate of the optimizer over time.
            initial_lr (float): the initial learning rate used by the optimizer.
        """
        raise NotImplemented   # pragma: no cover

    def get_images(self) -> Any:
        """Returns the stored images"""
        return self.images

    def get_optimizer(self) -> Any:
        """Returns the optimizer"""
        return self.optimizer

    def get_scheduler(self) -> Any:
        """Returns the scheduler"""
        return self.scheduler


class AllImagesStatsHolder:
    """
    Stores activation statistics for all image batches. It offers an organized mechanism for retaining mean,
    second-moment, and standard deviation statistics corresponding to the activations of each layer.

    Responsible for collecting and storing activation statistics across all batches of images.
    It stores a list 'batches_stats_holder_list' of 'BatchStatsHolder's. Each `BatchStatsHolder` instance in
    the `batches_stats_holder_list` is responsible for storing statistics for a specific batch, specified by "batch_index".
    """

    def __init__(self,
                 n_batches: int,
                 batch_size: int,
                 mean_axis: List):
        """
        Constructor for the AllImagesStatsHolder class.

        Args:
            n_batches (int): the number of batches.
            batch_size (int): the size of each batch.
            mean_axis (List): the axis along which to compute the mean.
        """
        self.mean_axis = mean_axis
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.data_bn_mean_all_batches_except_one = {}
        self.data_bn_second_moment_all_batches_except_one = {}
        self.bn_mean_all_batches = {}
        self.bn_second_moment_all_batches = {}
        self.batches_stats_holder_list = self.get_batches_stats_holder_list()

    def get_batches_stats_holder_list(self) -> list:
        """
        Get a list of batches stats holders.

        Returns:
            list: List of batches stats holders.
        """
        raise NotImplemented   # pragma: no cover

    def update_batch_stats(self,
                           batch_index: int,
                           input_imgs: Any,
                           activation_extractor: ActivationExtractor,
                           to_differentiate=False):
        """
        Update the batch statistics for a given batch.

        Args:
            batch_index (int): the index of the batch.
            input_imgs (Any): the input images for which to calculate the statistics.
            activation_extractor (ActivationsExtractor): the activation extractor object.
            to_differentiate (bool): a flag indicating whether to differentiate or not. Defaults to False.
        """
        self.batches_stats_holder_list[batch_index].clear()
        self.batches_stats_holder_list[batch_index].calc_bn_stats_from_activations(input_imgs=input_imgs,
                                                                                   activation_extractor=activation_extractor,
                                                                                   to_differentiate=to_differentiate)

    def get_stats(self,
                  batch_index: int,
                  layer_name: str) -> Tuple[Any, Any, Any]:
        """
        Get the statistics for a given batch and layer.

        Args:
            batch_index (int): the index of the batch.
            layer_name (str): the name of the layer.

        Returns:
            Tuple[Any, Any, Any]: the mean, second moment and std for the specified batch and layer.
        """
        mean = self.batches_stats_holder_list[batch_index].get_mean(layer_name)
        second_moment = self.batches_stats_holder_list[batch_index].get_second_moment(layer_name)
        var = self.batches_stats_holder_list[batch_index].get_var(layer_name)
        std = self.batches_stats_holder_list[batch_index].get_std(layer_name)
        return mean, second_moment, std


class BatchStatsHolder:
    """
    Stores activation statistics for a specific batch of images.
    This class provides a structured approach for managing mean, second-moment,
    and standard deviation statistics related to the activations of each layer
    for a particular batch of images.
    """

    def __init__(self,
                 mean_axis: List,
                 eps: float = 1e-6):
        """
        Constructor for the BatchStatsHolder class.

        Args:
            mean_axis (List): the axis along which to compute the mean.
            eps (float): a small value added to the denominator to avoid division by zero. Defaults to 1e-6.
        """
        self.eps = eps
        self.mean_axis = mean_axis
        self.bn_mean = {}
        self.bn_second_moment = {}

    def get_mean(self, bn_layer_name: str) -> Any:
        """
        Get the mean for the specified layer.

        Args:
            bn_layer_name (str): the name of the layer.

        Returns:
            Any: the mean for the specified layer.
        """
        return self.bn_mean.get(bn_layer_name)

    def get_second_moment(self, bn_layer_name: str) -> Any:
        """
        Get the second moment for the specified layer.

        Args:
            bn_layer_name (str): the name of the layer.

        Returns:
            Any: the second moment for the specified layer.
        """
        return self.bn_second_moment.get(bn_layer_name)

    def get_var(self, bn_layer_name: str) -> Any:
        """
        Calculate the variance for the specified layer.

        Args:
            bn_layer_name (str): the name of the layer.

        Returns:
            Any: the variance for the specified layer.
        """
        raise NotImplemented   # pragma: no cover

    def get_std(self, bn_layer_name: str) -> Any:
        """
        Calculate the standart deviation for the specified layer.

        Args:
            bn_layer_name (str): the name of the layer.

        Returns:
            Any: the variance for the specified layer.
        """
        raise NotImplemented   # pragma: no cover

    def update_layer_stats(self,
                           bn_layer_name: str,
                           mean: Any,
                           second_moment: Any):
        """
        Update the statistics for a layer.

        Args:
            bn_layer_name (str): the name of the layer.
            mean (Any): the mean value for the layer.
            second_moment (Any): the second moment value for the layer.
        """
        self.bn_mean.update({bn_layer_name: mean})
        self.bn_second_moment.update({bn_layer_name: second_moment})

    def calc_bn_stats_from_activations(self,
                                       input_imgs: Any,
                                       activation_extractor: ActivationExtractor,
                                       to_differentiate: bool):
        """
        Calculate statistics from the input images and activation extractor.

        Args:
            input_imgs (Any): the input images tensor for which to calculate the statistics.
            activation_extractor (ActivationExtractor): the activation extractor object.
            to_differentiate (bool): a flag indicating whether to differentiate or not.
        """
        raise NotImplemented   # pragma: no cover

    def clear(self):
        """Clear the statistics."""
        self.bn_mean.clear()
        self.bn_second_moment.clear()
        self.bn_mean = {}
        self.bn_second_moment = {}
