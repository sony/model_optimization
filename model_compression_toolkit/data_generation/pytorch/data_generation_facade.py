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
from functools import partial
from typing import Callable, Tuple, List, Any

from model_compression_toolkit.constants import FOUND_TORCH
from model_compression_toolkit.core.pytorch.utils import set_model
from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig, \
    BaseImagePipeline, ImageGranularity
from model_compression_toolkit.data_generation.pytorch.image_pipeline import PytorchImagePipeline, get_random_data
from model_compression_toolkit.data_generation.pytorch.model_info_exctractors import ActivationExtractor, \
    OrigBNStatsHolder
from model_compression_toolkit.data_generation.pytorch.optimization_utils import AllImagesOptimizationHandler
from model_compression_toolkit.logger import Logger

DEFAULT_INITIAL_LR = 5
DEFAULT_DATA_GEN_BS = 8


if FOUND_TORCH:
    import torch
    from torch.nn import Module
    from torch.optim import RAdam, Optimizer
    from torch.optim.lr_scheduler import ReduceLROnPlateau


    def get_pytorch_data_generation_config(
            n_iter: int,
            optimizer: Optimizer,
            scheduler: Any,
            scheduler_step_fn: Callable,
            data_gen_batch_size=DEFAULT_DATA_GEN_BS,
            initial_lr=DEFAULT_INITIAL_LR,
            bna_loss_fn: Callable = None,
            image_granularity=ImageGranularity.AllImages,
            image_pipeline: BaseImagePipeline = PytorchImagePipeline,
            image_padding: int = 0,
            layer_weighting_fn: Callable = None,
            activations_loss_fn: Callable = None,
            bn_layer_types: List = [torch.nn.BatchNorm2d],
            image_initialization_fn: Callable = None) -> DataGenerationConfig:
        """
        Function to create a DataGenerationConfig object with the specified configuration parameters.

        Args:
            n_iter (int): Number of iterations.
            optimizer (Optimizer): Optimizer.
            scheduler: Learning rate scheduler.
            scheduler_step_fn (Callable): Function to perform a scheduler step.
            data_gen_batch_size (int, optional): Batch size for data generation. Defaults to DEFAULT_DATA_GEN_BS.
            initial_lr (float, optional): Initial learning rate. Defaults to DEFAULT_INITIAL_LR.
            bna_loss_fn (Callable, optional): Loss function for BatchNorm alignment. Defaults to None.
            image_granularity (ImageGranularity, optional): Image granularity. Defaults to ImageGranularity.AllImages.
            image_pipeline (BaseImagePipeline, optional): Image pipeline class. Defaults to PytorchImagePipeline.
            image_padding (int, optional): Image padding size. Defaults to 0.
            layer_weighting_fn (Callable, optional): Function to compute layer weighting. Defaults to None.
            activations_loss_fn (Callable, optional): Function to compute activations loss. Defaults to None.
            bn_layer_types (List, optional): List of BatchNorm layer types. Defaults to [torch.nn.BatchNorm2d].
            image_initialization_fn (Callable, optional): Function for image initialization. Defaults to None.

        Returns:
            DataGenerationConfig: Data generation configuration object.
        """
        # Set default bna_loss_fn if not provided
        if bna_loss_fn is None:
            def bna_loss_fn(a, b):
                return torch.linalg.norm(a - b) ** 2 / b.size(0)

        # Set default layer_weighting_fn if not provided
        if layer_weighting_fn is None:
            def layer_weighting_fn(orig_bn_stats_holder: OrigBNStatsHolder):
                num_bn_layers = orig_bn_stats_holder.get_num_bn_layers()
                return {bn_layer_name: 1 / num_bn_layers for bn_layer_name in orig_bn_stats_holder.get_bn_layer_names()}

        # Set default activations_loss_fn if not provided
        if activations_loss_fn is None:
            def activations_loss_fn(imgs_input, output, activation_extractor, orig_bn_stats_holder):
                return 0

        # Create and return a DataGenerationConfig object with the specified parameters
        return DataGenerationConfig(
            n_iter=n_iter,
            optimizer=optimizer,
            scheduler=scheduler,
            data_gen_batch_size=data_gen_batch_size,
            initial_lr=initial_lr,
            scheduler_step_fn=scheduler_step_fn,
            bna_loss_fn=bna_loss_fn,
            image_granularity=image_granularity,
            image_pipeline=image_pipeline,
            image_padding=image_padding,
            layer_weighting_fn=layer_weighting_fn,
            activations_loss_fn=activations_loss_fn,
            bn_layer_types=bn_layer_types,
            image_initialization_fn=image_initialization_fn)


    def pytorch_data_generation_experimental(
            model: Module,
            n_images: int,
            output_image_size: Tuple,
            data_generation_config: DataGenerationConfig):
        """
        Function to perform data generation using the provided model and data generation configuration.

        Args:
            model (Module): PyTorch model to generate data for.
            n_images (int): Number of images to generate.
            output_image_size (Tuple): Size of the output images.
            data_generation_config (DataGenerationConfig): Configuration for data generation.

        Returns:
            DataLoader: Finalized data loader containing generated images.
        """
        # Set the current model
        set_model(model)

        # Create an image pipeline object using the specified output_image_size and image_padding
        image_pipeline = data_generation_config.image_pipeline(output_image_size, data_generation_config.image_padding)

        # Create an activation extractor object to extract activations from the model
        activation_extractor = ActivationExtractor(model, data_generation_config.bn_layer_types)

        # Create an orig_bn_stats_holder object to hold original BatchNorm statistics
        orig_bn_stats_holder = OrigBNStatsHolder(model, data_generation_config.bn_layer_types)

        # Initialize the dataset for data generation
        init_dataset = data_generation_config.image_initialization_fn(
            n_images=n_images,
            size=image_pipeline.get_image_input_size(),
            batch_size=data_generation_config.data_gen_batch_size)

        # Compute the layer weights based on orig_bn_stats_holder
        layer_weights = data_generation_config.layer_weighting_fn(orig_bn_stats_holder)

        # Create an AllImagesOptimizationHandler object for handling optimization
        all_imgs_opt_handler = AllImagesOptimizationHandler(
            init_dataset,
            data_generation_config,
            image_pipeline,
            activation_extractor)

        # Define the log intervals for iterations
        iter_log_interval = list(range(0, data_generation_config.n_iter , int(data_generation_config.n_iter / 10)))

        # Perform data generation iterations
        for i_ter in range(data_generation_config.n_iter):
            # Randomly reorder the batches
            all_imgs_opt_handler.random_batch_reorder()

            # Iterate over each batch
            for i_batch in range(all_imgs_opt_handler.n_batches):
                # Get the random batch index
                random_batch_index = all_imgs_opt_handler.get_random_batch_index(i_batch)

                # Get the images to optimize and the optimizer for the batch
                imgs_to_optimize = all_imgs_opt_handler.get_images_by_batch_index(random_batch_index)
                optimizer = all_imgs_opt_handler.get_optimizer_by_batch_index(random_batch_index)
                scheduler = all_imgs_opt_handler.get_scheduler_by_batch_index(random_batch_index)

                # Zero gradients
                optimizer.zero_grad()
                model.zero_grad()

                # Perform image input manipulation
                input_imgs = image_pipeline.image_input_manipulation(imgs_to_optimize)

                # Forward pass to extract activations
                output = activation_extractor.run_on_inputs(input_imgs)

                # Compute BatchNorm alignment loss
                bn_loss = all_imgs_opt_handler.compute_bn_loss(input_imgs=input_imgs,
                                                               batch_index=random_batch_index,
                                                               activation_extractor=activation_extractor,
                                                               orig_bn_stats_holder=orig_bn_stats_holder,
                                                               bn_loss_fn=data_generation_config.bna_loss_fn,
                                                               layer_weights=layer_weights)

                # Compute other activations losses
                other_losses = data_generation_config.activations_loss_fn(input_imgs,
                                                                          output,
                                                                          activation_extractor,
                                                                          orig_bn_stats_holder)

                # Compute total loss
                total_loss = bn_loss + other_losses
                # if i_ter in iter_log_interval and i_batch == (all_imgs_opt_handler.n_batches-1):
                #     print(f'iter {i_ter}-> bn loss: {bn_loss}, other losses: {other_losses}, total loss {total_loss}')

                # Backward pass
                total_loss.backward()

                # Update weights
                optimizer.step()

                # Perform scheduler step
                data_generation_config.scheduler_step_fn(scheduler, i_ter, total_loss.item())

                # Update the statistics based on the updated images
                final_imgs = image_pipeline.image_output_finalize(imgs_to_optimize)
                all_imgs_opt_handler.update_statistics(input_imgs=final_imgs,
                                              batch_index=random_batch_index,
                                              activation_extractor=activation_extractor)

                # Log iteration progress
                if i_ter in iter_log_interval and i_batch == (all_imgs_opt_handler.n_batches - 1):
                    Logger.info(f"Iteration {i_ter}/{data_generation_config.n_iter}: "
                                f"Total Loss: {total_loss.item():.5f}, "
                                f"BN Loss: {bn_loss.item():.5f}, "
                                f"Other Losses: {other_losses.item():.5f}")

        # Return the finalized data loader containing the generated images
        return all_imgs_opt_handler.get_finalized_data_loader()
else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def get_pytorch_data_generation_config(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using get_pytorch_data_generation_config. '
                        'Could not find torch package.')  # pragma: no cover


    def pytorch_data_generation_experimental(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using pytorch_data_generation_experimental. '
                        'Could not find the torch package.')  # pragma: no cover


if __name__ == '__main__':
    import torchvision
    from torchvision.models import ResNet18_Weights

    def scheduler_step_fn(scheduler, i_iter, loss_value):
        scheduler.step(loss_value)


    num_iterations=2000
    data_generation_config = get_pytorch_data_generation_config(
        n_iter=num_iterations,
        optimizer=RAdam,
        initial_lr=16,
        data_gen_batch_size=128,
        image_granularity=ImageGranularity.BatchWise,
        scheduler=partial(ReduceLROnPlateau, min_lr=1e-4, factor=0.5,  patience=int(num_iterations / 50)),
        scheduler_step_fn=scheduler_step_fn,
        image_padding=32,
        bn_layer_types=[torch.nn.BatchNorm2d],
        image_initialization_fn=get_random_data)

    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)


    pytorch_data_generation_experimental(model,
                                         n_images=128,
                                         output_image_size=224,
                                         data_generation_config=data_generation_config)
