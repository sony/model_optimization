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
from typing import Iterator, Any, Callable, Tuple, List, Dict

import numpy as np
import tensorflow as tf

from model_compression_toolkit.data_generation.common.constants import IMAGE_INPUT
from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig
from model_compression_toolkit.data_generation.common.enums import ImageGranularity
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline
from model_compression_toolkit.data_generation.common.optimization_utils import ImagesOptimizationHandler, \
    BatchOptimizationHolder, AllImagesStatsHolder, BatchStatsHolder
from model_compression_toolkit.data_generation.keras.constants import IMAGE_MIN_VAL, IMAGE_MAX_VAL, BATCH_AXIS, \
    H_AXIS, W_AXIS
from model_compression_toolkit.data_generation.keras.image_operations import create_valid_grid
from model_compression_toolkit.data_generation.keras.model_info_exctractors import KerasActivationExtractor, \
    KerasOriginalBNStatsHolder


class KerasImagesOptimizationHandler(ImagesOptimizationHandler):
    def __init__(self,
                 model: tf.keras.Model,
                 init_dataset: Iterator,
                 image_pipeline: BaseImagePipeline,
                 activation_extractor: KerasActivationExtractor,
                 scheduler: Any,
                 normalization_mean: List[float],
                 normalization_std: List[float],
                 data_generation_config: DataGenerationConfig,
                 orig_bn_stats_holder: KerasOriginalBNStatsHolder,
                 eps: float = 1e-6):
        """
        Constructor for the KerasImagesOptimizationHandler class.

        Args:
            model (tf.keras.Model): input model.
            init_dataset (DataLoader): The initial dataset used for image generation.
            image_pipeline (BaseImagePipeline): The image pipeline for processing images.
            activation_extractor (ActivationExtractor): Extractor for layer activations.
            scheduler (Any): The scheduler responsible for adjusting the learning rate of the optimizer over time.
            normalization_mean (List[float]): The mean values for image normalization.
            normalization_std (List[float]): The standard deviation values for image normalization.
            data_generation_config (DataGenerationConfig): Configuration for data generation.
            orig_bn_stats_holder (OriginalBNStatsHolder): Object to hold original BatchNorm statistics.
            eps (float): A small value added for numerical stability.
        """
        self.data_generation_config = data_generation_config
        self.optimizer = data_generation_config.optimizer
        self.orig_bn_stats_holder = orig_bn_stats_holder

        # Image valid grid, each image value can only be
        # IMAGE_MIN_VAL( default set to 0) - IMAGE_MAX_VAL ( default set to 255) before normalization
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.valid_grid = create_valid_grid(self.normalization_mean, self.normalization_std)

        super(KerasImagesOptimizationHandler, self).__init__(model=model,
                                                             data_gen_batch_size=
                                                             data_generation_config.data_gen_batch_size,
                                                             init_dataset=init_dataset,
                                                             optimizer=self.optimizer,
                                                             image_pipeline=image_pipeline,
                                                             activation_extractor=activation_extractor,
                                                             image_granularity=data_generation_config.image_granularity,
                                                             scheduler_step_fn=None,
                                                             scheduler=scheduler,
                                                             initial_lr=data_generation_config.initial_lr,
                                                             normalization_mean=self.normalization_mean,
                                                             normalization_std=self.normalization_std,
                                                             eps=eps)

        # Set the mean axis based on the image granularity
        if self.image_granularity == ImageGranularity.ImageWise:
            self.mean_axis = [H_AXIS, W_AXIS]
        else:
            self.mean_axis = [BATCH_AXIS, H_AXIS, W_AXIS]

        # Create BatchOptimizationHolder objects for each batch in the initial dataset
        self.batch_opt_holders_list = []
        self.batched_images_for_optimization = []
        for data_input in init_dataset:
            if isinstance(data_input, list):
                # This is the case in which the data loader holds both images and labels
                images, targets = data_input
            else:
                images = data_input

            # Define the imgs as tf.Variable
            batched_images = tf.Variable(initial_value=tf.zeros_like(images), trainable=True,
                                         constraint=lambda z: self.clip_and_reflect(z))
            batched_images.assign(value=images)

            self.batch_opt_holders_list.append(
                KerasBatchOptimizationHolder(
                    images=batched_images,
                    optimizer=self.optimizer,
                    scheduler=scheduler,
                    initial_lr=data_generation_config.initial_lr))
            self.batched_images_for_optimization.append(batched_images)

        self.n_batches = len(self.batch_opt_holders_list)
        self.random_batch_reorder()
        self.all_imgs_stats_holder = KerasAllImagesStatsHolder(n_batches=self.n_batches,
                                                               batch_size=self.batch_size,
                                                               mean_axis=self.mean_axis)
        if self.use_all_data_stats:
            for i_batch in range(self.n_batches):
                input_imgs = self.image_pipeline.image_output_finalize(images=self.get_images_by_batch_index(i_batch))
                output = activation_extractor.run_model(inputs=input_imgs)
                self.all_imgs_stats_holder.update_batch_stats(batch_index=i_batch,
                                                              input_imgs=input_imgs,
                                                              activation_extractor=activation_extractor)

    def clip_and_reflect(self,
                         z: tf.Tensor) -> tf.Tensor:
        """
        Clips and optionally reflects the input tensor `z` channel-wise based on the valid value range.

        Args:
            z (tf.Tensor): Input tensor to be clipped and reflected.

        Returns:
            tf.Tensor: Clipped and reflected tensor.
        """
        images = z.numpy()
        for i_ch in range(len(self.valid_grid)):
            # Clip the values of the channel within the valid range.
            clamp = tf.clip_by_value(t=z[:, :, :, i_ch], clip_value_min=self.valid_grid[i_ch][0],
                                     clip_value_max=self.valid_grid[i_ch][1])
        # Assign the clipped reflected values back to `z`.
        z.assign(images)
        return z

    def get_layer_accumulated_stats(self, layer_name: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get the accumulated activation statistics for a layer.

        Args:
            layer_name (str): the name of the layer.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The averaged activation statistics (mean and second moment) on all the
            batches for the specified layer.
        """
        total_mean, total_second_moment = 0, 0
        for i_batch in range(self.n_batches):
            mean, second_moment = self.all_imgs_stats_holder.get_stats(batch_index=i_batch,
                                                                       layer_name=layer_name)
            total_mean += mean
            total_second_moment += second_moment

        total_mean /= self.n_batches
        total_second_moment /= self.n_batches
        return tf.expand_dims(input=total_mean, axis=0), tf.expand_dims(input=total_second_moment, axis=0)

    def compute_bn_loss(self,
                        input_imgs: tf.Tensor,
                        batch_index: int,
                        activation_extractor: KerasActivationExtractor,
                        bn_layer_weights: Dict,
                        bn_alignment_loss_fn: Callable) -> tf.Tensor:
        """
        Compute the batch norm alignment loss.

        Args:
            input_imgs (Any): The input images.
            batch_index (int): The index of the batch.
            activation_extractor (ActivationExtractor): Extractor for layer activations.
            bn_layer_weights (Dict): Weights to multiply the loss for each layer.
            bn_alignment_loss_fn (Callable): Function to compute BatchNorm alignment loss.

        Returns:
            Any: The computed batch norm alignment loss.
        """

        self.all_imgs_stats_holder.update_batch_stats(batch_index=batch_index,
                                                      input_imgs=input_imgs,
                                                      activation_extractor=activation_extractor)

        # Initialize variables for accumulating mean and variance differences
        total_bn_loss, total_mean_diff, total_var_diff = 0, 0, 0

        for layer_name in self.orig_bn_stats_holder.get_bn_layer_names():
            # Get the layer weight for the current BN layer
            layer_weight = bn_layer_weights.get(layer_name)

            # Get the mean and variance from the original BN statistics
            bn_layer_mean = self.orig_bn_stats_holder.get_mean(bn_layer_name=layer_name)
            bn_layer_var = self.orig_bn_stats_holder.get_var(bn_layer_name=layer_name)

            # Get the mean and variance from the current batch's statistics
            if self.use_all_data_stats:
                # If using all data statistics, retrieve the accumulated statistics from all batches
                imgs_layer_mean, imgs_layer_var = self.get_layer_accumulated_stats(layer_name=layer_name)
            else:
                # Otherwise, retrieve the statistics from the current batch
                imgs_layer_mean, imgs_layer_var = (
                    self.all_imgs_stats_holder.get_stats(batch_index=batch_index, layer_name=layer_name))

            bn_layer_std = tf.sqrt(bn_layer_var + self.eps)
            imgs_layer_std = tf.sqrt(imgs_layer_var + self.eps)

            total_bn_loss += layer_weight * bn_alignment_loss_fn(bn_mean=bn_layer_mean,
                                                                 input_mean=imgs_layer_mean,
                                                                 bn_std=bn_layer_std,
                                                                 input_std=imgs_layer_std)
        return total_bn_loss

    def update_statistics(self,
                          input_imgs: tf.Tensor,
                          batch_index: int,
                          activation_extractor: KerasActivationExtractor):
        """
        Update the statistics for the images at the specified batch index.

        Args:
            input_imgs (Any): The input images.
            batch_index (int): The index of the batch.
            activation_extractor (ActivationExtractor): Extractor for layer activations.
        """
        self.all_imgs_stats_holder.update_batch_stats(
            batch_index=batch_index,
            input_imgs=input_imgs,
            activation_extractor=activation_extractor)
        self.batched_images_for_optimization[batch_index] = input_imgs

    def optimization_step(self,
                          batch_index: int,
                          images: tf.Tensor,
                          gradients: tf.Tensor,
                          loss: tf.Tensor,
                          i_iter: int):
        """
        Perform an optimization step.

        Args:
            batch_index (int): Index of the batch.
            images (tf.Tensor): The images to optimize for the batch.
            gradients (List[tf.Tensor]): The gradients calculated for the images.
            loss (tf.Tensor): Loss value.
            i_iter (int): Current optimization iteration.
        """
        # Get optimizer and scheduler for the specific batch index
        optimizer = self.get_optimizer_by_batch_index(batch_index=batch_index)
        scheduler = self.get_scheduler_by_batch_index(batch_index=batch_index)

        # Update images
        optimizer.apply_gradients(zip(gradients, [images]))

        # Perform scheduler step
        scheduler.on_epoch_end(epoch=i_iter, loss=tf.reduce_mean(loss))

    def get_finilized_data_loader(self) -> np.ndarray:
        """
        Create and return a ndarray of the generated images.

        Returns:
            List[np.ndarray]: List containing the generated images.
        """
        finalized_images = []

        # Iterate over each batch
        for i_batch in range(self.n_batches):
            # Retrieve the images for the current batch
            batch_imgs = self.get_images_by_batch_index(i_batch)

            # Apply the image_pipeline's image_output_finalize method to finalize the batch of images
            finalized_batch = self.image_pipeline.image_output_finalize(batch_imgs).numpy()
            finalized_images += np.split(finalized_batch, indices_or_sections=self.batch_size, axis=BATCH_AXIS)
        return finalized_images


class KerasBatchOptimizationHolder(BatchOptimizationHolder):
    """
    An extension of BatchOptimizationHolder specifically for Keras models.
    Holds optimization parameters for a batch of images.

    This class acts as a container for optimization-related parameters specific to a batch of images. It does not
    directly manage or handle the optimization process itself but rather holds the necessary components for
    optimization, including images, optimizer and scheduler.
    """

    def __init__(self,
                 images: tf.Variable,
                 optimizer: Any,
                 scheduler: Any,
                 initial_lr: float):
        """
        Constructor for the KerasBatchOptimizationHolder class.

        Args:
            images (tf.Tensor): A tensor containing the input images.
            optimizer (Any): An optimizer responsible for updating the image parameters during optimization.
            scheduler (Any): A scheduler responsible for adjusting the learning rate of the optimizer over time.
            initial_lr (float): The initial learning rate used by the optimizer.
        """
        self.images = images
        self.optimizer = optimizer(learning_rate=initial_lr)
        self.scheduler = scheduler(optimizer=self.optimizer)


class KerasAllImagesStatsHolder(AllImagesStatsHolder):
    """
    An extension of AllImagesStatsHolder specifically for Keras models.
    Stores activation statistics for all image batches. It offers an organized mechanism for retaining mean,
    second-moment, and standard deviation statistics corresponding to the activations of each layer.

    Responsible for collecting and storing activation statistics across all batches of images.
    It stores a list 'batches_stats_holder_list' of 'BatchStatsHolder's. Each `BatchStatsHolder` instance in
    the `batches_stats_holder_list` is responsible for storing statistics for a specific batch,
    specified by "batch_index".
    """

    def get_batches_stats_holder_list(self) -> List[BatchStatsHolder]:
        """
        Get a list of BatchStatsHolder objects.

        Returns:
            List[BatchStatsHolder]: A list of BatchStatsHolder objects.
        """
        return [KerasBatchStatsHolder(self.mean_axis) for _ in range(self.n_batches)]

    def update_batch_stats(self,
                           batch_index: int,
                           input_imgs: tf.Tensor,
                           activation_extractor: KerasActivationExtractor):
        """
        Update the batch statistics for a given batch.

        Args:
            batch_index (int): The index of the batch.
            input_imgs (tf.Tensor): The input images for which to calculate the statistics.
            activation_extractor (ActivationsExtractor): The activation extractor object.
        """
        self.batches_stats_holder_list[batch_index].clear()
        self.batches_stats_holder_list[batch_index].calc_bn_stats_from_activations(input_imgs=input_imgs,
                                                                                   activation_extractor=
                                                                                   activation_extractor)

    def get_stats(self,
                  batch_index: int,
                  layer_name: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get the statistics for a given batch and layer.

        Args:
            batch_index (int): The index of the batch.
            layer_name (str): The name of the layer.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The mean and second moment for the specified batch and layer.
        """
        mean = self.batches_stats_holder_list[batch_index].get_mean(bn_layer_name=layer_name)
        second_moment = self.batches_stats_holder_list[batch_index].get_second_moment(bn_layer_name=layer_name)
        return mean, second_moment


class KerasBatchStatsHolder(BatchStatsHolder):
    """
    An extension of BatchStatsHolder specifically for Keras models.
    Stores activation statistics for a specific batch of images.
    This class provides a structured approach for managing mean, second-moment,
    and standard deviation statistics related to the activations of each layer
    for a particular batch of images.
    """

    def calc_bn_stats_from_activations(self,
                                       input_imgs: tf.Tensor,
                                       activation_extractor: KerasActivationExtractor):
        """
        Calculate and update statistics (mean, second-moment) per layer, using the input images and activations.

        This function calculates and updates the mean and second-moment statistics for layers before batch normalization
        using the corresponding activations. The calculated statistics are used to align the batch normalization
        statistics stored in the original model during the data generation process.

        Args:
            input_imgs (tf.Tensor): the input images tensor for which to calculate the statistics.
            activation_extractor (KerasActivationExtractor): the activation extractor object.
        """
        imgs_mean, imgs_second_moment = tf.nn.moments(x=input_imgs, axes=self.mean_axis, keepdims=False)
        self.update_layer_stats(IMAGE_INPUT, imgs_mean, imgs_second_moment)

        # Extract statistics of intermediate convolution outputs before the BatchNorm layers
        for bn_layer_name in activation_extractor.get_extractor_layer_names():
            bn_input_activations = activation_extractor.get_layer_input_activation(layer_name=bn_layer_name)
            collected_mean, collected_second_moment = tf.nn.moments(x=bn_input_activations['input_data'],
                                                                    axes=self.mean_axis, keepdims=False)
            self.update_layer_stats(bn_layer_name=bn_layer_name,
                                    mean=collected_mean,
                                    second_moment=collected_second_moment)
