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
import time
from typing import Callable, Tuple, List
from tqdm import tqdm

from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig, \
    ImageGranularity
from model_compression_toolkit.data_generation.common.enums import BatchNormAlignemntLossType, DataInitType, \
    BNLayerWeightingType, ImagePipelineType, ImageNormalizationType, SchedulerType, OptimizerType, OutputLossType

if FOUND_TF:
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization
    from model_compression_toolkit.data_generation.keras.image_pipeline import (image_pipeline_dict,
                                                                                image_normalization_dict)
    from model_compression_toolkit.data_generation.keras.model_info_exctractors import (KerasActivationExtractor,
                                                                                        KerasOriginalBNStatsHolder)
    from model_compression_toolkit.data_generation.keras.optimization_functions.batchnorm_alignment_functions import \
        bn_alignment_loss_function_dict
    from model_compression_toolkit.data_generation.keras.optimization_functions.bn_layer_weighting_functions import \
        bn_layer_weighting_function_dict
    from model_compression_toolkit.data_generation.keras.optimization_functions.image_initilization import \
        image_initilization_function_dict
    from model_compression_toolkit.data_generation.keras.optimization_utils import KerasImagesOptimizationHandler, \
        KerasBatchStatsHolder
    from model_compression_toolkit.data_generation.keras.optimization_functions.output_loss_functions import \
        output_loss_function_dict
    from model_compression_toolkit.data_generation.keras.optimization_functions.optimizer_types import optimizers_dict
    from model_compression_toolkit.data_generation.keras.optimization_functions.scheduler_step_functions import \
        scheduler_step_function_dict
    from model_compression_toolkit.data_generation.keras.constants import DEFAULT_N_ITER, DEFAULT_DATA_GEN_BS, \
        DEFAULT_INITIAL_LR, DEFAULT_OUTPUT_LOSS_MULTIPLIER


    # Function to create a DataGenerationConfig object with the specified configuration parameters for Tensorflow
    def get_tensorflow_data_generation_config(
            n_iter: int = DEFAULT_N_ITER,
            optimizer_type: OptimizerType = OptimizerType.ADAM,
            data_gen_batch_size: int = DEFAULT_DATA_GEN_BS,
            initial_lr: float = DEFAULT_INITIAL_LR,
            output_loss_multiplier: float = DEFAULT_OUTPUT_LOSS_MULTIPLIER,
            scheduler_type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU,
            bn_alignment_loss_type: BatchNormAlignemntLossType = BatchNormAlignemntLossType.L2_SQUARE,
            output_loss_type: OutputLossType = OutputLossType.MARGINAL_MIN_MAX_DIFF,
            data_init_type: DataInitType = DataInitType.Gaussian,
            layer_weighting_type: BNLayerWeightingType = BNLayerWeightingType.AVERAGE,
            image_granularity: ImageGranularity = ImageGranularity.BatchWise,
            image_pipeline_type: ImagePipelineType = ImagePipelineType.RANDOM_CROP_FLIP,
            image_normalization_type: ImageNormalizationType = ImageNormalizationType.MOBILENET,
            extra_pixels: int = 0,
            activations_loss_fn: Callable = None,
            bn_layer_types: List = [BatchNormalization],
            clip_images: bool = True,
            reflection: bool = True,
    ) -> DataGenerationConfig:
        """
        Function to create a DataGenerationConfig object with the specified configuration parameters.

        Args:
            n_iter (int): Number of iterations for the data generation process.
            optimizer_type (OptimizerType): The optimizer type to use for the data generation process.
            data_gen_batch_size (int): Batch size for data generation.
            initial_lr (float): Initial learning rate for the optimizer.
            output_loss_multiplier (float): Multiplier for the output loss during optimization.
            scheduler_type (SchedulerType): The type of scheduler to use.
            bn_alignment_loss_type (BatchNormAlignemntLossType): The type of BatchNorm alignment loss to use.
            output_loss_type (OutputLossType): The type of output loss to use.
            data_init_type (DataInitType): The type of data initialization to use.
            layer_weighting_type (BNLayerWeightingType): The type of layer weighting to use.
            image_granularity (ImageGranularity): The granularity of the images for optimization.
            image_pipeline_type (ImagePipelineType): The type of image pipeline to use.
            image_normalization_type (ImageNormalizationType): The type of image normalization to use.
            extra_pixels (int): Extra pixels add to the input image size. Defaults to 0.
            activations_loss_fn (Callable): Activation loss function to use during optimization.
            bn_layer_types (List): List of BatchNorm layer types to be considered for data generation.
            clip_images (bool): Whether to clip images during optimization.
            reflection (bool): Whether to use reflection during optimization.

        Returns:
            DataGenerationConfig: Data generation configuration object.
        """
        # Get the optimizer class corresponding to the specified type
        optimizer = optimizers_dict.get(optimizer_type)

        # Check if the optimizer type is valid
        if optimizer is None:
            Logger.exception(
                f'Invalid optimizer_type {optimizer_type}.'
                f'Please choose one of {OptimizerType.get_values()}')

        # Create and return a DataGenerationConfig object with the specified parameters
        return DataGenerationConfig(
            n_iter=n_iter,
            optimizer=optimizer,
            data_gen_batch_size=data_gen_batch_size,
            initial_lr=initial_lr,
            scheduler_type=scheduler_type,
            bn_alignment_loss_type=bn_alignment_loss_type,
            output_loss_type=output_loss_type,
            data_init_type=data_init_type,
            layer_weighting_type=layer_weighting_type,
            image_granularity=image_granularity,
            image_pipeline_type=image_pipeline_type,
            image_normalization_type=image_normalization_type,
            extra_pixels=extra_pixels,
            activations_loss_fn=activations_loss_fn,
            bn_layer_types=bn_layer_types,
            clip_images=clip_images,
            reflection=reflection,
            output_loss_multiplier=output_loss_multiplier)


    def tensorflow_data_generation_experimental(
            model: tf.keras.Model,
            n_images: int,
            output_image_size: Tuple,
            data_generation_config: DataGenerationConfig) -> tf.Tensor:
        """
        Function to perform data generation using the provided model and data generation configuration.

        Args:
            model (Model): Keras model to generate data for.
            n_images (int): Number of images to generate.
            output_image_size (Tuple): Size of the output images.
            data_generation_config (DataGenerationConfig): Configuration for data generation.

        Returns:
            Tensor: Finalized generated images.
        """

        # Get the image pipeline class corresponding to the specified type
        image_pipeline = (
            image_pipeline_dict.get(data_generation_config.image_pipeline_type)(
                output_image_size=output_image_size,
                extra_pixels=data_generation_config.extra_pixels))

        # Check if the image pipeline type is valid
        if image_pipeline is None:
            Logger.exception(
                f'Invalid image_pipeline_type {data_generation_config.image_pipeline_type}.'
                f'Please choose one of {ImagePipelineType.get_values()}')

        # Get the normalization function corresponding to the specified type
        normalization = image_normalization_dict.get(data_generation_config.image_normalization_type)

        # Check if the image normalization type is valid
        if normalization is None:
            Logger.exception(
                f'Invalid image_normalization_type {data_generation_config.image_normalization_type}.'
                f'Please choose one of {ImageNormalizationType.get_values()}')

        # Get the layer weighting function corresponding to the specified type
        bn_layer_weighting_fn = bn_layer_weighting_function_dict.get(data_generation_config.layer_weighting_type)

        if bn_layer_weighting_fn is None:
            Logger.exception(
                f'Invalid layer_weighting_type {data_generation_config.layer_weighting_type}.'
                f'Please choose one of {BNLayerWeightingType.get_values()}')

        # Get the image initialization function corresponding to the specified type
        image_initialization_fn = image_initilization_function_dict.get(data_generation_config.data_init_type)

        # Check if the data initialization type is valid
        if image_initialization_fn is None:
            Logger.exception(
                f'Invalid data_init_type {data_generation_config.data_init_type}.'
                f'Please choose one of {DataInitType.get_values()}')

        # Get the scheduler functions corresponding to the specified scheduler type
        scheduler_get_fn = scheduler_step_function_dict.get(data_generation_config.scheduler_type)

        # Check if the scheduler type is valid
        if scheduler_get_fn is None:
            Logger.exception(
                f'Invalid scheduler_type {data_generation_config.scheduler_type}.'
                f'Please choose one of {SchedulerType.get_values()}')

        # Create a scheduler object with the specified number of iterations
        scheduler = scheduler_get_fn(n_iter=data_generation_config.n_iter,
                                     initial_lr=data_generation_config.initial_lr)

        # Get the BatchNorm alignment loss function corresponding to the specified type
        bn_alignment_loss_fn = bn_alignment_loss_function_dict.get(data_generation_config.bn_alignment_loss_type)

        # Check if the BatchNorm alignment loss type is valid
        if bn_alignment_loss_fn is None:
            Logger.exception(
                f'Invalid bn_alignment_loss_type {data_generation_config.bn_alignment_loss_type}.'
                f'Please choose one of {BatchNormAlignemntLossType.get_values()}')

        # Get the output loss function corresponding to the specified type
        output_loss_fn = output_loss_function_dict.get(data_generation_config.output_loss_type)

        # Check if the output loss type is valid
        if output_loss_fn is None:
            Logger.exception(
                f'Invalid output_loss_type {data_generation_config.output_loss_type}.'
                f'Please choose one of {OutputLossType.get_values()}')

        # Set the model to eval mode
        model.trainable = False

        # Initialize the dataset for data generation
        num_batches, init_dataset = image_initialization_fn(
            n_images=n_images,
            image_size=image_pipeline.get_image_input_size(),
            batch_size=data_generation_config.data_gen_batch_size)

        # Create an activation extractor object to extract activations from the model
        activation_extractor = KerasActivationExtractor(model=model,
                                                        layer_types_to_extract_inputs=
                                                        data_generation_config.bn_layer_types,
                                                        image_granularity=data_generation_config.image_granularity,
                                                        image_input_manipulation=
                                                        image_pipeline.image_input_manipulation)

        # Create an orig_bn_stats_holder object to hold original BatchNorm statistics
        orig_bn_stats_holder = KerasOriginalBNStatsHolder(model=model,
                                                          bn_layer_types=data_generation_config.bn_layer_types)
        if orig_bn_stats_holder.get_num_bn_layers() == 0:
            Logger.exception(
                f'Data generation requires a model with at least one BatchNorm layer.')

        # Create an ImagesOptimizationHandler object for handling optimization
        all_imgs_opt_handler = KerasImagesOptimizationHandler(
            init_dataset=init_dataset,
            data_generation_config=data_generation_config,
            image_pipeline=image_pipeline,
            activation_extractor=activation_extractor,
            scheduler=scheduler,
            model=model,
            bn_layer_weights_fn=bn_layer_weighting_fn,
            bn_loss_fn=bn_alignment_loss_fn,
            output_loss_fn=output_loss_fn,
            orig_bn_stats_holder=orig_bn_stats_holder)

        # Get the current time to measure the total time taken
        total_time = time.time()

        # Create a tqdm progress bar for iterating over data_generation_config.n_iter iterations
        ibar = tqdm(range(data_generation_config.n_iter))

        # Perform data generation iterations
        for i_ter in ibar:

            # Randomly reorder the batches
            all_imgs_opt_handler.random_batch_reorder()

            # Iterate over each batch
            for i_batch in range(all_imgs_opt_handler.n_batches):
                # Get the random batch index
                random_batch_index = all_imgs_opt_handler.get_random_batch_index(index=i_batch)

                # Get the images to optimize and the optimizer for the batch
                imgs_to_optimize = all_imgs_opt_handler.get_images_by_batch_index(batch_index=random_batch_index)

                # Get the batch stats holder for the batch
                batch_stats_holder = all_imgs_opt_handler.all_imgs_stats_holder.batches_stats_holder_list[
                    random_batch_index]

                # Compute the gradients and the loss for the batch
                gradients, total_loss, bn_loss, output_loss = keras_compute_grads(imgs_to_optimize=imgs_to_optimize,
                                                                                  batch_stats_holder=batch_stats_holder,
                                                                                  activation_extractor=
                                                                                  activation_extractor,
                                                                                  all_imgs_opt_handler=
                                                                                  all_imgs_opt_handler)

                # Perform optimization step
                all_imgs_opt_handler.optimization_step(batch_index=random_batch_index,
                                                       images=imgs_to_optimize,
                                                       gradients=gradients,
                                                       loss=total_loss,
                                                       i_ter=i_ter)

                # Update the statistics based on the updated images
                if all_imgs_opt_handler.use_all_data_stats:
                    final_imgs = image_pipeline.image_output_finalize(images=imgs_to_optimize)
                    all_imgs_opt_handler.update_statistics(input_imgs=final_imgs,
                                                           batch_index=random_batch_index,
                                                           batch_stats_holder=batch_stats_holder,
                                                           activation_extractor=activation_extractor)

            ibar.set_description(f"Total Loss: {total_loss.numpy().mean().item():.5f}, "
                                 f"BN Loss: {bn_loss.numpy().mean().item():.5f}, "
                                 f"Output Loss: {output_loss.numpy().mean().item():.5f}")

        # Return a list containing the finalized generated images
        finalized_imgs = all_imgs_opt_handler.get_finilized_data_loader()
        Logger.info(f'Total time to generate {len(finalized_imgs)} images (seconds): {int(time.time() - total_time)}')
        return finalized_imgs


    # Compute the gradients and the loss for the batch
    @tf.function
    def keras_compute_grads(imgs_to_optimize: tf.Tensor,
                            batch_stats_holder: KerasBatchStatsHolder,
                            activation_extractor: KerasActivationExtractor,
                            all_imgs_opt_handler: KerasImagesOptimizationHandler) -> (
            Tuple)[List[tf.Tensor], tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        This function run part of the training step, calculating the losses and the gradients for the batch,
        wrapped by a tf.function for acceleration.

        Args:
            imgs_to_optimize (tf.Tensor): The images to optimize for the batch.
            batch_stats_holder (KerasBatchStatsHolder): Stats holder for the batch.
            activation_extractor (KerasActivationExtractor): extractor for layer activations.
            all_imgs_opt_handler (KerasImagesOptimizationHandler): Handles the images optimization process.

        Returns:
    Returns:
        Tuple[List[tf.Tensor], tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing:
        - The gradients calculated for the images
        - Total loss
        - BN loss
        - Output loss
        """
        # Create a gradient tape to compute gradients
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # Watch the images to be optimized to compute gradients.
            tape.watch(imgs_to_optimize)

            # Perform image input manipulation
            input_imgs = all_imgs_opt_handler.image_pipeline.image_input_manipulation(images=imgs_to_optimize)

            # Forward pass to extract activations
            output = activation_extractor.run_model(inputs=input_imgs)

            # Compute BatchNorm alignment loss
            bn_loss = all_imgs_opt_handler.compute_bn_loss(input_imgs=input_imgs,
                                                           batch_stats_holder=batch_stats_holder,
                                                           activation_extractor=activation_extractor)

            # Compute output loss
            output_loss = all_imgs_opt_handler.compute_output_loss(output_imgs=output,
                                                                   activation_extractor=activation_extractor,
                                                                   tape=tape)

            # Compute total loss
            total_loss = bn_loss + output_loss

            # Get the trainable variables
            variables = [imgs_to_optimize]

        # Compute gradients of the total loss with respect to the images
        gradients = tape.gradient(total_loss, variables)

        # Return the computed gradients and individual loss components
        return gradients, total_loss, bn_loss, output_loss


else:
    def get_tensorflow_data_generation_config(*args, **kwargs):
        Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                        'when using get_tensorflow_data_generation_config. '
                        'Could not find Tensorflow package.')  # pragma: no cover


    def tensorflow_data_generation_experimental(*args, **kwargs):
        Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                        'when using pytorch_data_generation_experimental. '
                        'Could not find Tensorflow package.')  # pragma: no cover
