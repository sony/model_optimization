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
from typing import Callable, Tuple, List

from torch import Tensor

from model_compression_toolkit.constants import FOUND_TORCH
from model_compression_toolkit.core.pytorch.utils import set_model
from model_compression_toolkit.data_generation.common.data_generation import data_generation
from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig
from model_compression_toolkit.data_generation.common.enums import ImageGranularity, SchedulerType, \
    BatchNormAlignemntLossType, DataInitType, BNLayerWeightingType, ImagePipelineType, ImageNormalizationType, \
    OutputLossType
from model_compression_toolkit.data_generation.pytorch.image_pipeline import image_pipeline_dict, \
    image_normalization_dict
from model_compression_toolkit.data_generation.pytorch.model_info_exctractors import PytorchActivationExtractor, \
    PytorchOriginalBNStatsHolder
from model_compression_toolkit.data_generation.pytorch.optimization_functions.batchnorm_alignment_functions import \
    bn_alignment_loss_function_dict
from model_compression_toolkit.data_generation.pytorch.optimization_functions.image_initilization import \
    image_initilization_function_dict
from model_compression_toolkit.data_generation.pytorch.optimization_functions.bn_layer_weighting_functions import \
    bn_layer_weighting_function_dict
from model_compression_toolkit.data_generation.pytorch.optimization_functions.output_loss_functions import \
    output_loss_function_dict
from model_compression_toolkit.data_generation.pytorch.optimization_functions.scheduler_step_functions import \
    scheduler_step_function_dict
from model_compression_toolkit.data_generation.pytorch.optimization_utils import PytorchImagesOptimizationHandler, \
    DatasetFromList
from model_compression_toolkit.logger import Logger

# Define default values for various configuration parameters
DEFAULT_INITIAL_LR = 5
DEFAULT_OUTPUT_LOSS_MULTIPLIER = 0.0001
DEFAULT_DATA_GEN_BS = 32
DEFAULT_N_ITER = 500

if FOUND_TORCH:
    # Importing necessary libraries
    import torch
    from torch.nn import Module
    from torch.optim import RAdam, Optimizer
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    # Function to create a DataGenerationConfig object with the specified configuration parameters for Pytorch
    def get_pytorch_data_generation_config(
            n_iter: int = DEFAULT_N_ITER,
            optimizer: Optimizer = RAdam,
            data_gen_batch_size=DEFAULT_DATA_GEN_BS,
            initial_lr=DEFAULT_INITIAL_LR,
            output_loss_multiplier=DEFAULT_OUTPUT_LOSS_MULTIPLIER,
            scheduler_type: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU,
            bn_alignment_loss_type: BatchNormAlignemntLossType = BatchNormAlignemntLossType.L2_SQUARE,
            output_loss_type: OutputLossType = OutputLossType.MIN_MAX_DIFF,
            data_init_type: DataInitType = DataInitType.Diverse,
            layer_weighting_type: BNLayerWeightingType = BNLayerWeightingType.AVERAGE,
            image_granularity=ImageGranularity.AllImages,
            image_pipeline_type: ImagePipelineType = ImagePipelineType.RANDOM_CROP,
            image_normalization_type: ImageNormalizationType = ImageNormalizationType.TORCHVISION,
            extra_pixels: int = 0,
            activations_loss_fn: Callable = None,
            bn_layer_types: List = [torch.nn.BatchNorm2d],
            clip_images: bool = True,
            reflection: bool = True,
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
            extra_pixels (int): Extra pixels add to the input image size. Defaults to 0.
            activations_loss_fn (Callable): Activation loss function to use during optimization.
            bn_layer_types (List): List of BatchNorm layer types to be considered for data generation.
            clip_images (bool): Whether to clip images during optimization.
            reflection (bool): Whether to use reflection during optimization.


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
            activations_loss_fn=activations_loss_fn,
            bn_layer_types=bn_layer_types,
            clip_images=clip_images,
            reflection=reflection
        )


    def pytorch_data_generation_experimental(
            model: Module,
            n_images: int,
            output_image_size: Tuple,
            data_generation_config: DataGenerationConfig) -> List[Tensor]:
        """
        Function to perform data generation using the provided model and data generation configuration.

        Args:
            model (Module): PyTorch model to generate data for.
            n_images (int): Number of images to generate.
            output_image_size (Tuple): Size of the output images.
            data_generation_config (DataGenerationConfig): Configuration for data generation.

        Returns:
            List[Tensor]: Finalized list containing generated images.
        """
        # Get the image pipeline class corresponding to the specified type
        image_pipeline = image_pipeline_dict.get(data_generation_config.image_pipeline_type)(output_image_size,
                                                                                             data_generation_config.extra_pixels)
        # Check if the image pipeline type is valid
        if image_pipeline is None:
            Logger.exception(
                f'Invalid image_pipeline_type {data_generation_config.image_pipeline_type}. Please choose one of {ImagePipelineType.get_values()}')

        # Get the normalization function corresponding to the specified type
        normalization = image_normalization_dict.get(data_generation_config.image_normalization_type)

        # Check if the image normalization type is valid
        if normalization is None:
            Logger.exception(
                f'Invalid image_normalization_type {data_generation_config.image_normalization_type}. Please choose one of {ImageNormalizationType.get_values()}')

        # Get the layer weighting function corresponding to the specified type
        bn_layer_weighting_fn = bn_layer_weighting_function_dict.get(data_generation_config.layer_weighting_type)

        # Check if the layer weighting type is valid
        if bn_layer_weighting_fn is None:
            Logger.exception(
                f'Invalid layer_weighting_type {data_generation_config.layer_weighting_type}. Please choose one of {BNLayerWeightingType.get_values()}')

        # Get the image initialization function corresponding to the specified type
        image_initialization_fn = image_initilization_function_dict.get(data_generation_config.data_init_type)

        # Check if the data initialization type is valid
        if image_initialization_fn is None:
            Logger.exception(
                f'Invalid data_init_type {data_generation_config.data_init_type}. Please choose one of {DataInitType.get_values()}')

        # Get the scheduler functions corresponding to the specified scheduler type
        scheduler_get_fn, scheduler_step_fn = scheduler_step_function_dict.get(data_generation_config.scheduler_type)

        # Check if the scheduler type is valid
        if scheduler_get_fn is None or scheduler_step_fn is None:
            Logger.exception(
                f'Invalid scheduler_type {data_generation_config.scheduler_type}. Please choose one of {SchedulerType.get_values()}')

        # Create a scheduler object with the specified number of iterations
        scheduler = scheduler_get_fn(data_generation_config.n_iter)

        # Get the BatchNorm alignment loss function corresponding to the specified type
        bn_alignment_loss_fn = bn_alignment_loss_function_dict.get(data_generation_config.bn_alignment_loss_type)

        # Check if the BatchNorm alignment loss type is valid
        if bn_alignment_loss_fn is None:
            Logger.exception(
                f'Invalid bn_alignment_loss_type {data_generation_config.bn_alignment_loss_type}. Please choose one of {BatchNormAlignemntLossType.get_values()}')

        # Get the output loss function corresponding to the specified type
        output_loss_fn = output_loss_function_dict.get(data_generation_config.output_loss_type)

        # Check if the output loss type is valid
        if bn_alignment_loss_fn is None:
            Logger.exception(
                f'Invalid output_loss_type {data_generation_config.output_loss_type}. Please choose one of {OutputLossType.get_values()}')

        # Set the current model
        set_model(model)

        # Initialize the dataset for data generation
        init_dataset = image_initialization_fn(
            n_images=n_images,
            size=image_pipeline.get_image_input_size(),
            batch_size=data_generation_config.data_gen_batch_size)

        # Create an activation extractor object to extract activations from the model
        activation_extractor = PytorchActivationExtractor(model, data_generation_config.bn_layer_types)

        # Create an orig_bn_stats_holder object to hold original BatchNorm statistics
        orig_bn_stats_holder = PytorchOriginalBNStatsHolder(model, data_generation_config.bn_layer_types)
        if orig_bn_stats_holder.get_num_bn_layers() == 0:
            Logger.exception(
                f'Data generation requires a model with at least one Batch Norm layer.')


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
                                                                   clip_images=data_generation_config.clip_images,
                                                                   reflection=data_generation_config.reflection)

        # Perform data generation and obtain a list of generated images
        generated_images_list = data_generation(data_generation_config=data_generation_config,
                                                activation_extractor=activation_extractor,
                                                orig_bn_stats_holder=orig_bn_stats_holder,
                                                all_imgs_opt_handler=all_imgs_opt_handler,
                                                image_pipeline=image_pipeline,
                                                bn_layer_weighting_fn=bn_layer_weighting_fn,
                                                bn_alignment_loss_fn=bn_alignment_loss_fn,
                                                output_loss_fn=output_loss_fn,
                                                output_loss_multiplier=data_generation_config.output_loss_multiplier,
                                                )
        # Return the list of finalized generated images
        return generated_images_list
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
