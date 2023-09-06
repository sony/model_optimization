from typing import Iterator, Any, Callable, Tuple, Optional, List

import numpy as np
import tensorflow as tf

from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline
from model_compression_toolkit.data_generation.common.optimization_utils import ImagesOptimizationHandler, \
    BatchOptimizationHolder, AllImagesStatsHolder, BatchStatsHolder
from model_compression_toolkit.data_generation.keras.constants import IMAGE_INPUT, IMAGE_MIN_VAL, IMAGE_MAX_VAL
from model_compression_toolkit.data_generation.keras.image_pipeline import image_normalization_dict
from model_compression_toolkit.data_generation.keras.model_info_exctractors import KerasActivationExtractor, \
    KerasOriginalBNStatsHolder


class KerasImagesOptimizationHandler(ImagesOptimizationHandler):
    def __init__(self,
                 model: tf.keras.Model,
                 init_dataset: Iterator,
                 image_pipeline: BaseImagePipeline,
                 activation_extractor: KerasActivationExtractor,
                 scheduler: Any,
                 data_generation_config: DataGenerationConfig,
                 bn_layer_weights_fn: Callable,
                 bn_loss_fn: Callable,
                 output_loss_fn: Callable,
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
            data_generation_config (DataGenerationConfig): Configuration for data generation.
            bn_layer_weights_fn (Callable): Function to compute layer weighting for the BatchNorm alignment loss .
            bn_loss_fn (Callable): Function to compute BatchNorm alignment loss.
            output_loss_fn (Callable): Function to compute output loss.
            orig_bn_stats_holder (OriginalBNStatsHolder): Object to hold original BatchNorm statistics.
            eps (float): A small value added for numerical stability.
        """
        self.data_generation_config = data_generation_config
        self.optimizer = data_generation_config.optimizer
        self.orig_bn_stats_holder = orig_bn_stats_holder

        # Set the batch normalization bn layer weights function, bn loss function,
        # output loss function, and output loss multiplier.
        self.bn_layer_weights_fn = bn_layer_weights_fn
        self.bn_loss_fn = bn_loss_fn
        self.output_loss_fn = output_loss_fn
        self.output_loss_multiplier = data_generation_config.output_loss_multiplier

        # Image valid grid, each image value can only be 0 - 255 before normalization
        self.image_normalization = image_normalization_dict[data_generation_config.image_normalization_type]
        self.normalization_mean = self.image_normalization[0]
        self.normalization_std = self.image_normalization[1]
        self.valid_grid = []
        for i, (mean, var) in enumerate(zip(self.image_normalization[0], self.image_normalization[1])):
            min_val = (IMAGE_MIN_VAL - mean) / var
            max_val = (IMAGE_MAX_VAL - mean) / var
            self.valid_grid.append((min_val, max_val))

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
                                                             clip_images=data_generation_config.clip_images,
                                                             reflection=data_generation_config.reflection,
                                                             eps=eps)

        # Set the mean axis based on the image granularity
        self.mean_axis = activation_extractor.mean_axis

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
                batch_stats_holder = self.all_imgs_stats_holder.batches_stats_holder_list[i_batch]
                self.all_imgs_stats_holder.update_batch_stats(batch_stats_holder=batch_stats_holder,
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
        if self.clip_images:
            images = z.numpy()
            for i_ch in range(len(self.valid_grid)):
                # Clip the values of the channel within the valid range.
                clamp = tf.clip_by_value(t=z[:, :, :, i_ch], clip_value_min=self.valid_grid[i_ch][0],
                                         clip_value_max=self.valid_grid[i_ch][1])
                if self.reflection:
                    # Reflect the values.
                    images[:, :, :, i_ch] = 2 * clamp - z[:, :, :, i_ch]
                else:
                    images[:, :, :, i_ch] = clamp
            # Assign the clipped reflected values back to `z`.
            z.assign(images)
            return z
        else:
            return z

    def get_accumulated_stats_per_layer(self, layer_name: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get the accumulated activation statistics for a layer.

        Args:
            layer_name (str): the name of the layer.

        Returns: Tuple[tf.Tensor, tf.Tensor]: The averaged activation statistics (mean and second moment) on all the
        batches for the specified layer.
        """
        total_mean, total_second_moment = 0, 0
        for i_batch in range(self.n_batches):
            batch_stats_holder = self.all_imgs_stats_holder.batches_stats_holder_list[i_batch]
            mean, second_moment = self.all_imgs_stats_holder.get_stats(batch_stats_holder=batch_stats_holder,
                                                                       layer_name=layer_name)
            total_mean += mean
            total_second_moment += second_moment

        total_mean /= self.n_batches
        total_second_moment /= self.n_batches
        return tf.expand_dims(input=total_mean, axis=0), tf.expand_dims(input=total_second_moment, axis=0)

    def compute_bn_loss(self,
                        input_imgs: tf.Tensor,
                        batch_stats_holder: BatchStatsHolder,
                        activation_extractor: KerasActivationExtractor) -> tf.Tensor:
        """
        Compute the batch norm alignment loss.

        Args:
            input_imgs (Any): the input images.
            batch_stats_holder (BatchStatsHolder): batch stats holder.
            activation_extractor (ActivationExtractor): extractor for layer activations.

        Returns:
            Any: the computed batch norm alignment loss.
        """

        self.all_imgs_stats_holder.update_batch_stats(batch_stats_holder=batch_stats_holder,
                                                      input_imgs=input_imgs,
                                                      activation_extractor=activation_extractor)

        # Initialize variables for accumulating mean and variance differences
        total_bn_loss, total_mean_diff, total_var_diff = 0, 0, 0

        # Iterate over each BN layer
        layer_weights = self.bn_layer_weights_fn(orig_bn_stats_holder=self.orig_bn_stats_holder)

        for layer_name in self.orig_bn_stats_holder.get_bn_layer_names():
            # Get the layer weight for the current BN layer
            layer_weight = layer_weights.get(layer_name)

            # Get the mean and variance from the original BN statistics
            bn_layer_mean = self.orig_bn_stats_holder.get_mean(bn_layer_name=layer_name)
            bn_layer_var = self.orig_bn_stats_holder.get_var(bn_layer_name=layer_name)

            # Get the mean and variance from the current batch's statistics
            if self.use_all_data_stats:
                # If using all data statistics, retrieve the accumulated statistics from all batches
                imgs_layer_mean, imgs_layer_var = self.get_accumulated_stats_per_layer(layer_name=layer_name)
            else:
                # Otherwise, retrieve the statistics from the current batch
                imgs_layer_mean, imgs_layer_var = (
                    self.all_imgs_stats_holder.get_stats(batch_stats_holder=batch_stats_holder, layer_name=layer_name))

            bn_layer_std = tf.sqrt(bn_layer_var + self.eps)
            imgs_layer_std = tf.sqrt(imgs_layer_var + self.eps)

            total_bn_loss += layer_weight * self.bn_loss_fn(bn_mean=bn_layer_mean,
                                                            input_mean=imgs_layer_mean,
                                                            bn_std=bn_layer_std,
                                                            input_std=imgs_layer_std)
        return total_bn_loss

    def compute_output_loss(self,
                            output_imgs: tf.Tensor,
                            activation_extractor: KerasActivationExtractor,
                            tape: tf.GradientTape
                            ) -> tf.Tensor:
        # If output_loss_multiplier is zero return 0
        return self.output_loss_multiplier * self.output_loss_fn(
            output_imgs=output_imgs,
            activation_extractor=activation_extractor,
            tape=tape,
            weights_last_layer=
            self.get_model_output_weight_norm(model=self.model, activation_extractor=activation_extractor)) \
            if self.output_loss_multiplier > 0 else tf.zeros(1)

    def get_model_output_weight_norm(self,
                                     model: tf.keras.Model,
                                     activation_extractor: KerasActivationExtractor) -> Optional[tf.Tensor]:
        """
        Get the weight tensor of the last Dense or Conv2D layer in the model.

        Args:
            model (Model): The model to retrieve the weight tensor from.
            activation_extractor (ActivationExtractor): Extractor for layer activations.

        Returns:
            Optional[tf.Tensor]: The weight tensor of the last linear layer, or None if not found.
        """
        # Iterate through the layers in reverse order.
        for layer in reversed(model.layers):
            if isinstance(layer, activation_extractor.linear_layers):
                # Return the weight tensor of the layer.
                return layer.weights[0] if layer.weights else None

        # If no suitable layer is found, return None.
        return None

    def update_statistics(self,
                          input_imgs: tf.Tensor,
                          batch_index: int,
                          batch_stats_holder: BatchStatsHolder,
                          activation_extractor: KerasActivationExtractor):
        """
        Update the statistics for the images at the specified batch index.

        Args:
            input_imgs (Any): the input images.
            batch_index (int): the index of the batch.
            batch_stats_holder (BatchStatsHolder): batch stats holder.
            activation_extractor (ActivationExtractor): extractor for layer activations.
        """
        self.all_imgs_stats_holder.update_batch_stats(
            batch_stats_holder=batch_stats_holder,
            input_imgs=input_imgs,
            activation_extractor=activation_extractor)
        self.batched_images_for_optimization[batch_index] = input_imgs

    def optimization_step(self,
                          batch_index: int,
                          images: tf.Tensor,
                          gradients: tf.Tensor,
                          loss: tf.Tensor,
                          i_ter: int):
        """
        Perform an optimization step.

        Args:
            batch_index (int): Index of the batch.
            images (tf.Tensor): The images to optimize for the batch.
            gradients (List[tf.Tensor]): The gradients calculated for the images.
            loss (tf.Tensor): Loss value.
            i_ter (int): Current optimization iteration.
        """
        # Get optimizer and scheduler for the specific batch index
        optimizer = self.get_optimizer_by_batch_index(batch_index=batch_index)
        scheduler = self.get_scheduler_by_batch_index(batch_index=batch_index)

        # Update images
        optimizer.apply_gradients(zip(gradients, [images]))

        # Perform scheduler step
        scheduler.on_epoch_end(epoch=i_ter, loss=tf.reduce_mean(loss))

    def get_finilized_data_loader(self) -> np.ndarray:
        """
        Create and return a ndarray of the generated images.

        Returns:
            np.ndarray: the generated images.
        """
        finalized_images = []

        # Iterate over each batch
        for i_batch in range(self.n_batches):
            # Retrieve the images for the current batch
            batch_imgs = self.get_images_by_batch_index(i_batch)

            # Apply the image_pipeline's image_output_finalize method to finalize the batch of images
            finalized_batch = self.image_pipeline.image_output_finalize(batch_imgs)
            finalized_images.append(finalized_batch)
        return tf.concat(finalized_images, axis=0).numpy()


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
        self.optimizer = optimizer(lr=initial_lr)
        self.scheduler = scheduler(optim_lr=self.optimizer)


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
                           batch_stats_holder: BatchStatsHolder,
                           input_imgs: tf.Tensor,
                           activation_extractor: KerasActivationExtractor):
        """
        Update the batch statistics for a given batch.

        Args:
            batch_stats_holder (BatchStatsHolder): batch stats holder.
            input_imgs (tf.Tensor): the input images for which to calculate the statistics.
            activation_extractor (ActivationsExtractor): the activation extractor object.
        """
        batch_stats_holder.clear()
        batch_stats_holder.calc_bn_stats_from_activations(input_imgs=input_imgs,
                                                          activation_extractor=activation_extractor)

    def get_stats(self,
                  batch_stats_holder: BatchStatsHolder,
                  layer_name: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get the statistics for a given batch and layer.

        Args:
            batch_stats_holder (BatchStatsHolder): batch stats holder.
            layer_name (str): the name of the layer.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: the mean and second moment for the specified batch and layer.
        """
        mean = batch_stats_holder.get_mean(bn_layer_name=layer_name)
        second_moment = batch_stats_holder.get_second_moment(bn_layer_name=layer_name)
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
        for bn_layer_name in activation_extractor.get_bn_layer_names():
            bn_input_activations = activation_extractor.get_activation(layer_name=bn_layer_name)
            collected_mean, collected_second_moment = tf.nn.moments(x=bn_input_activations['input_data'],
                                                                    axes=self.mean_axis, keepdims=False)
            self.update_layer_stats(bn_layer_name=bn_layer_name,
                                    mean=collected_mean,
                                    second_moment=collected_second_moment)
