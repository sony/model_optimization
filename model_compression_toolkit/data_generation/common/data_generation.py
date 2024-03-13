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
# Import required modules and classes
from typing import Any, Tuple, Dict, Callable, List

from model_compression_toolkit.data_generation.common.data_generation_config import DataGenerationConfig
from model_compression_toolkit.data_generation.common.enums import ImagePipelineType, ImageNormalizationType, \
    BNLayerWeightingType, DataInitType, BatchNormAlignemntLossType, OutputLossType
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline
from model_compression_toolkit.logger import Logger


def get_data_generation_classes(
        data_generation_config: DataGenerationConfig,
        output_image_size: Tuple,
        n_images: int,
        image_pipeline_dict: Dict,
        image_normalization_dict: Dict,
        bn_layer_weighting_function_dict: Dict,
        image_initialization_function_dict: Dict,
        bn_alignment_loss_function_dict: Dict,
        output_loss_function_dict: Dict) \
        -> Tuple[BaseImagePipeline, List[List[float]], Callable, Callable, Callable, Any]:
    """
    Function to create a DataGenerationConfig object with the specified configuration parameters.

    Args:
        data_generation_config (DataGenerationConfig): Configuration for data generation.
        output_image_size (Tuple): The desired output image size.
        n_images (int): The number of random samples.
        image_pipeline_dict (Dict): Dictionary mapping ImagePipelineType to corresponding image pipeline classes.
        image_normalization_dict (Dict): Dictionary mapping ImageNormalizationType to corresponding
        normalization values.
        bn_layer_weighting_function_dict (Dict): Dictionary of layer weighting functions.
        image_initialization_function_dict (Dict): Dictionary of image initialization functions.
        bn_alignment_loss_function_dict (Dict): Dictionary of batch normalization alignment loss functions.
        output_loss_function_dict (Dict): Dictionary of output loss functions.

    Returns:
        image_pipeline (BaseImagePipeline): The image pipeline for processing images during optimization.
        normalization (List[List[float]]): The image normalization values for processing images during optimization.
        bn_layer_weighting_fn (Callable): Function to compute layer weighting for the BatchNorm alignment loss.
        bn_alignment_loss_fn (Callable): Function to compute BatchNorm alignment loss.
        output_loss_fn (Callable): Function to compute output loss.
        init_dataset (Any): The initial dataset used for image generation.
    """
    # Get the image pipeline class corresponding to the specified type
    image_pipeline = (
        image_pipeline_dict.get(data_generation_config.image_pipeline_type)(
            output_image_size=output_image_size,
            extra_pixels=data_generation_config.extra_pixels))

    # Check if the image pipeline type is valid
    if image_pipeline is None:
        Logger.critical(
            f'Invalid image_pipeline_type {data_generation_config.image_pipeline_type}. '
            f'Please select one from {ImagePipelineType.get_values()}.')

    # Get the normalization values corresponding to the specified type
    normalization = image_normalization_dict.get(data_generation_config.image_normalization_type)

    # Check if the image normalization type is valid
    if normalization is None:
        Logger.critical(
            f'Invalid image_normalization_type {data_generation_config.image_normalization_type}. '
            f'Please select one from {ImageNormalizationType.get_values()}.')

    # Get the layer weighting function corresponding to the specified type
    bn_layer_weighting_fn = bn_layer_weighting_function_dict.get(data_generation_config.layer_weighting_type)

    if bn_layer_weighting_fn is None:
        Logger.critical(
            f'Invalid layer_weighting_type {data_generation_config.layer_weighting_type}. '
            f'Please select one from {BNLayerWeightingType.get_values()}.')

    # Get the image initialization function corresponding to the specified type
    image_initialization_fn = image_initialization_function_dict.get(data_generation_config.data_init_type)

    # Check if the data initialization type is valid
    if image_initialization_fn is None:
        Logger.critical(
            f'Invalid data_init_type {data_generation_config.data_init_type}. '
            f'Please select one from {DataInitType.get_values()}.')

    # Get the BatchNorm alignment loss function corresponding to the specified type
    bn_alignment_loss_fn = bn_alignment_loss_function_dict.get(data_generation_config.bn_alignment_loss_type)

    # Check if the BatchNorm alignment loss type is valid
    if bn_alignment_loss_fn is None:
        Logger.critical(
            f'Invalid bn_alignment_loss_type {data_generation_config.bn_alignment_loss_type}. '
            f'Please select one from {BatchNormAlignemntLossType.get_values()}.')

    # Get the output loss function corresponding to the specified type
    output_loss_fn = output_loss_function_dict.get(data_generation_config.output_loss_type)

    # Check if the output loss type is valid
    if output_loss_fn is None:
        Logger.critical(
            f'Invalid output_loss_type {data_generation_config.output_loss_type}. '
            f'Please select one from {OutputLossType.get_values()}.')

    # Initialize the dataset for data generation
    init_dataset = image_initialization_fn(
        n_images=n_images,
        size=image_pipeline.get_image_input_size(),
        batch_size=data_generation_config.data_gen_batch_size)

    return image_pipeline, normalization, bn_layer_weighting_fn, bn_alignment_loss_fn, output_loss_fn, init_dataset
