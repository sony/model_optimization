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
from typing import Callable, Any, Tuple, List, Union

from tqdm import tqdm

from model_compression_toolkit.verify_packages import FOUND_TORCHVISION, FOUND_TORCH
from model_compression_toolkit.core.pytorch.utils import set_model
from model_compression_toolkit.data_generation.common.constants import DEFAULT_N_ITER, DEFAULT_DATA_GEN_BS
from model_compression_toolkit.data_generation.common.data_generation import get_data_generation_classes
from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig
from model_compression_toolkit.data_generation.common.enums import ImageGranularity, SchedulerType, \
    BatchNormAlignemntLossType, DataInitType, BNLayerWeightingType, ImagePipelineType, ImageNormalizationType, \
    OutputLossType
from model_compression_toolkit.data_generation.common.image_pipeline import image_normalization_dict
from model_compression_toolkit.data_generation.pytorch.constants import DEFAULT_PYTORCH_INITIAL_LR, \
    DEFAULT_PYTORCH_BN_LAYER_TYPES, DEFAULT_PYTORCH_LAST_LAYER_TYPES, DEFAULT_PYTORCH_EXTRA_PIXELS, \
    DEFAULT_PYTORCH_OUTPUT_LOSS_MULTIPLIER
from model_compression_toolkit.data_generation.pytorch.image_pipeline import image_pipeline_dict, \
    BaseImagePipeline
from model_compression_toolkit.data_generation.pytorch.model_info_exctractors import PytorchActivationExtractor, \
    PytorchOriginalBNStatsHolder
from model_compression_toolkit.data_generation.pytorch.optimization_functions.batchnorm_alignment_functions import \
    bn_alignment_loss_function_dict
from model_compression_toolkit.data_generation.pytorch.optimization_functions.bn_layer_weighting_functions import \
    bn_layer_weighting_function_dict
from model_compression_toolkit.data_generation.pytorch.optimization_functions.image_initilization import \
    image_initialization_function_dict
from model_compression_toolkit.data_generation.pytorch.optimization_functions.output_loss_functions import \
    output_loss_function_dict
from model_compression_toolkit.data_generation.pytorch.optimization_functions.scheduler_step_functions import \
    scheduler_step_function_dict
from model_compression_toolkit.data_generation.pytorch.optimization_utils import PytorchImagesOptimizationHandler
from model_compression_toolkit.logger import Logger

if FOUND_TORCH and FOUND_TORCHVISION:
    # Importing necessary libraries
    import torch
    from torch import Tensor
    from torch.nn import Module
    from torch.optim import RAdam, Optimizer
    from torch.fx import symbolic_trace
    from torch.cuda.amp import autocast

    from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device

    # Function to create a DataGenerationConfig object with the specified configuration parameters for Pytorch
    def get_pytorch_data_generation_config(
            n_iter: int = DEFAULT_N_ITER,
            optimizer: Optimizer = RAdam,
            data_gen_batch_size=DEFAULT_DATA_GEN_BS,
            initial_lr=DEFAULT_PYTORCH_INITIAL_LR,
            output_loss_multiplier=DEFAULT_PYTORCH_OUTPUT_LOSS_MULTIPLIER,
            scheduler_type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU_WITH_RESET,
            bn_alignment_loss_type: BatchNormAlignemntLossType = BatchNormAlignemntLossType.L2_SQUARE,
            output_loss_type: OutputLossType = OutputLossType.NEGATIVE_MIN_MAX_DIFF,
            data_init_type: DataInitType = DataInitType.Gaussian,
            layer_weighting_type: BNLayerWeightingType = BNLayerWeightingType.AVERAGE,
            image_granularity=ImageGranularity.AllImages,
            image_pipeline_type: ImagePipelineType = ImagePipelineType.SMOOTHING_AND_AUGMENTATION,
            image_normalization_type: ImageNormalizationType = ImageNormalizationType.TORCHVISION,
            extra_pixels: Union[int, Tuple[int, int]] = DEFAULT_PYTORCH_EXTRA_PIXELS,
            bn_layer_types: List = DEFAULT_PYTORCH_BN_LAYER_TYPES,
            last_layer_types: List = DEFAULT_PYTORCH_LAST_LAYER_TYPES,
            image_clipping: bool = True,
    ) -> DataGenerationConfig:
        """
        Function to create a DataGenerationConfig object with the specified configuration parameters.

        Args:
            n_iter (int): Number of iterations for the data generation process.
            optimizer (Optimizer): The optimizer to use for the data generation process.
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
            extra_pixels (Union[int, Tuple[int, int]]): Extra pixels to add to the input image size. Defaults to 0.
            bn_layer_types (List): List of BatchNorm layer types to be considered for data generation.
            last_layer_types (List): List of layer types to be considered for the output loss.
            image_clipping (bool): Whether to clip images during optimization.


        Returns:
            DataGenerationConfig: Data generation configuration object.
        """

        # Create and return a DataGenerationConfig object with the specified parameters
        return DataGenerationConfig(
            n_iter=n_iter,
            optimizer=optimizer,
            data_gen_batch_size=data_gen_batch_size,
            initial_lr=initial_lr,
            output_loss_multiplier=output_loss_multiplier,
            scheduler_type=scheduler_type,
            bn_alignment_loss_type=bn_alignment_loss_type,
            output_loss_type=output_loss_type,
            data_init_type=data_init_type,
            layer_weighting_type=layer_weighting_type,
            image_granularity=image_granularity,
            image_pipeline_type=image_pipeline_type,
            image_normalization_type=image_normalization_type,
            extra_pixels=extra_pixels,
            bn_layer_types=bn_layer_types,
            last_layer_types=last_layer_types,
            image_clipping=image_clipping,
        )


    def pytorch_data_generation_experimental(
            model: Module,
            n_images: int,
            output_image_size: Union[int, Tuple[int, int]],
            data_generation_config: DataGenerationConfig) -> List[Tensor]:
        """
        Function to perform data generation using the provided model and data generation configuration.

        Args:
            model (Module): PyTorch model to generate data for.
            n_images (int): Number of images to generate.
            output_image_size (Union[int, Tuple[int, int]]): The hight and width size of the output images.
            data_generation_config (DataGenerationConfig): Configuration for data generation.

        Returns:
            List[Tensor]: Finalized list containing generated images.

        Examples:

            In this example, we'll walk through generating images using a simple PyTorch model and a data generation configuration. The process involves creating a model, setting up a data generation configuration, and finally generating images with specified parameters.

            Start by importing the Model Compression Toolkit (MCT), PyTorch, and some modules from `torch.nn`:

            >>> import model_compression_toolkit as mct
            >>> import torch.nn as nn
            >>> from torch.nn import Conv2d, BatchNorm2d, Flatten, Linear

            Next, define a simple PyTorch model:

            >>> model = nn.Sequential(nn.Conv2d(3, 2, 3), nn.BatchNorm2d(2), nn.Flatten(), nn.Linear(2*6*6, 10))

            Configure the data generation process using `get_pytorch_data_generation_config`. This function allows customization of the data generation process. For simplicity, this example sets the number of iterations (`n_iter`) to 1 and the batch size (`data_gen_batch_size`) to 2.

            >>> config = mct.data_generation.get_pytorch_data_generation_config(n_iter=1, data_gen_batch_size=2)

            Finally, use the `pytorch_data_generation_experimental` function to generate images based on the model and data generation configuration.
            Notice that this function is experimental and may change in future versions of MCT.
            The `n_images` parameter specifies the number of images to generate, and `output_image_size` sets the size of the generated images.

            >>> generated_images = mct.data_generation.pytorch_data_generation_experimental(model=model, n_images=4, output_image_size=8, data_generation_config=config)

            The generated images can then be used for various purposes, such as data-free quantization.

        """

        Logger.warning(f"pytorch_data_generation_experimental is experimental "
                       f"and is subject to future changes."
                       f"If you encounter an issue, please open an issue in our GitHub "
                       f"project https://github.com/sony/model_optimization")

        # get the model device
        device = get_working_device()

        # get a static graph representation of the model using torch.fx
        fx_model = symbolic_trace(model)
		
		# Get Data Generation functions and classes
        image_pipeline, normalization, bn_layer_weighting_fn, bn_alignment_loss_fn, output_loss_fn, \
            init_dataset = get_data_generation_classes(data_generation_config=data_generation_config,
                                                       output_image_size=output_image_size,
                                                       n_images=n_images,
                                                       image_pipeline_dict=image_pipeline_dict,
                                                       image_normalization_dict=image_normalization_dict,
                                                       bn_layer_weighting_function_dict=
                                                       bn_layer_weighting_function_dict,
                                                       image_initialization_function_dict=
                                                       image_initialization_function_dict,
                                                       bn_alignment_loss_function_dict=bn_alignment_loss_function_dict,
                                                       output_loss_function_dict=output_loss_function_dict)

        # Get the scheduler functions corresponding to the specified scheduler type
        scheduler_get_fn, scheduler_step_fn = scheduler_step_function_dict.get(data_generation_config.scheduler_type)

        # Check if the scheduler type is valid
        if scheduler_get_fn is None or scheduler_step_fn is None:
            Logger.critical(f'Invalid scheduler_type {data_generation_config.scheduler_type}. '
                            f'Please select one from {SchedulerType.get_values()}.') # pragma: no cover

        # Create a scheduler object with the specified number of iterations
        scheduler = scheduler_get_fn(data_generation_config.n_iter)

        # Set the current model
        set_model(model)

        # Create an activation extractor object to extract activations from the model
        activation_extractor = PytorchActivationExtractor(
            model,
            fx_model,
            data_generation_config.bn_layer_types,
            data_generation_config.last_layer_types)

        # Create an orig_bn_stats_holder object to hold original BatchNorm statistics
        orig_bn_stats_holder = PytorchOriginalBNStatsHolder(model, data_generation_config.bn_layer_types)
        if orig_bn_stats_holder.get_num_bn_layers() == 0:
            Logger.critical(
                f'Data generation requires a model with at least one BatchNorm layer.') # pragma: no cover

        # Create an ImagesOptimizationHandler object for handling optimization
        all_imgs_opt_handler = PytorchImagesOptimizationHandler(model=model,
                                                                data_gen_batch_size=data_generation_config.data_gen_batch_size,
                                                                init_dataset=init_dataset,
                                                                optimizer=data_generation_config.optimizer,
                                                                image_pipeline=image_pipeline,
                                                                activation_extractor=activation_extractor,
                                                                image_granularity=data_generation_config.image_granularity,
                                                                scheduler_step_fn=scheduler_step_fn,
                                                                scheduler=scheduler,
                                                                initial_lr=data_generation_config.initial_lr,
                                                                normalization_mean=normalization[0],
                                                                normalization_std=normalization[1],
                                                                device=device)

        # Perform data generation and obtain a list of generated images
        generated_images_list = data_generation(
            data_generation_config=data_generation_config,
            activation_extractor=activation_extractor,
            orig_bn_stats_holder=orig_bn_stats_holder,
            all_imgs_opt_handler=all_imgs_opt_handler,
            image_pipeline=image_pipeline,
            bn_layer_weighting_fn=bn_layer_weighting_fn,
            bn_alignment_loss_fn=bn_alignment_loss_fn,
            output_loss_fn=output_loss_fn,
            output_loss_multiplier=data_generation_config.output_loss_multiplier,
            device=device,
        )
        # Return the list of finalized generated images
        return generated_images_list


    def data_generation(
            data_generation_config: DataGenerationConfig,
            activation_extractor: PytorchActivationExtractor,
            orig_bn_stats_holder: PytorchOriginalBNStatsHolder,
            all_imgs_opt_handler: PytorchImagesOptimizationHandler,
            image_pipeline: BaseImagePipeline,
            bn_layer_weighting_fn: Callable,
            bn_alignment_loss_fn: Callable,
            output_loss_fn: Callable,
            output_loss_multiplier: float,
            device: torch.device
    ) -> List[Any]:
        """
        Function to perform data generation using the provided model and data generation configuration.

        Args:
            data_generation_config (DataGenerationConfig): Configuration for data generation.
            activation_extractor (PytorchActivationExtractor): The activation extractor for the model.
            orig_bn_stats_holder (PytorchOriginalBNStatsHolder): Object to hold original BatchNorm statistics.
            all_imgs_opt_handler (PytorchImagesOptimizationHandler): Handles the images optimization process.
            image_pipeline (Callable): Callable image pipeline for image manipulation.
            bn_layer_weighting_fn (Callable): Function to compute layer weighting for the BatchNorm alignment loss .
            bn_alignment_loss_fn (Callable): Function to compute BatchNorm alignment loss.
            output_loss_fn (Callable): Function to compute output loss.
            output_loss_multiplier (float): Multiplier for the output loss.
            device (torch.device): The current device set for PyTorch operations.

        Returns:
            List: Finalized list containing generated images.
        """
        # Get the current time to measure the total time taken
        total_time = time.time()

        # Create a tqdm progress bar for iterating over data_generation_config.n_iter iterations
        ibar = tqdm(range(data_generation_config.n_iter))

        # Perform data generation iterations
        for i_iter in ibar:

            # Randomly reorder the batches
            all_imgs_opt_handler.random_batch_reorder()

            # Iterate over each batch
            for i_batch in range(all_imgs_opt_handler.n_batches):
                # Get the random batch index
                random_batch_index = all_imgs_opt_handler.get_random_batch_index(i_batch)

                # Get the images to optimize and the optimizer for the batch
                imgs_to_optimize = all_imgs_opt_handler.get_images_by_batch_index(random_batch_index)

                # Zero gradients
                all_imgs_opt_handler.zero_grad(random_batch_index)

                # Perform image input manipulation
                input_imgs = image_pipeline.image_input_manipulation(imgs_to_optimize)

                # Forward pass to extract activations
                with autocast():
                    output = activation_extractor.run_model(input_imgs)

                # Compute the layer weights based on orig_bn_stats_holder
                bn_layer_weights = bn_layer_weighting_fn(orig_bn_stats_holder, activation_extractor, i_iter, data_generation_config.n_iter)

                # Compute BatchNorm alignment loss
                bn_loss = all_imgs_opt_handler.compute_bn_loss(input_imgs=input_imgs,
                                                               batch_index=random_batch_index,
                                                               activation_extractor=activation_extractor,
                                                               orig_bn_stats_holder=orig_bn_stats_holder,
                                                               bn_alignment_loss_fn=bn_alignment_loss_fn,
                                                               bn_layer_weights=bn_layer_weights)

                # Compute output loss
                output_loss = output_loss_fn(
                    model_outputs=output,
                    activation_extractor=activation_extractor,
                    device=device)

                # Compute total loss
                total_loss = bn_loss + output_loss_multiplier * output_loss

                # Perform optimiztion step
                all_imgs_opt_handler.optimization_step(random_batch_index, total_loss, i_iter)

                # Update the statistics based on the updated images
                if all_imgs_opt_handler.use_all_data_stats:
                    with autocast():
                        final_imgs = image_pipeline.image_output_finalize(imgs_to_optimize)
                        all_imgs_opt_handler.update_statistics(input_imgs=final_imgs,
                                                               batch_index=random_batch_index,
                                                               activation_extractor=activation_extractor)

            ibar.set_description(f"Total Loss: {total_loss.item():.5f}, "
                                 f"BN Loss: {bn_loss.item():.5f}, "
                                 f"Output Loss: {output_loss.item():.5f}")

        # Return a list containing the finalized generated images
        finalized_imgs = all_imgs_opt_handler.get_finalized_images()
        Logger.info(f'Total time to generate {len(finalized_imgs)} images (seconds): {int(time.time() - total_time)}')
        Logger.info(f'Final Loss: Total {total_loss.item()}, BN loss {bn_loss.item()}, Output loss {output_loss.item()}')
        return finalized_imgs
else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def get_pytorch_data_generation_config(*args, **kwargs):
        msg = f"PyTorch and torchvision must be installed to use get_pytorch_data_generation_config. " + ("" if FOUND_TORCH else "The 'torch' package is missing. ") + ("" if FOUND_TORCHVISION else "The 'torchvision' package is missing. ") # pragma: no cover
        Logger.critical(msg)  # pragma: no cover

    def pytorch_data_generation_experimental(*args, **kwargs):
        msg = f"PyTorch and torchvision must be installed to use pytorch_data_generation_experimental. " + ("" if FOUND_TORCH else "The 'torch' package is missing. ") + ("" if FOUND_TORCHVISION else "The 'torchvision' package is missing. ") # pragma: no cover
        Logger.critical(msg)  # pragma: no cover