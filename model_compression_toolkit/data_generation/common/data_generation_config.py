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
from typing import Callable, Any, List

from model_compression_toolkit.data_generation.common.enums import SchedularType, BatchNormAlignemntLossType, \
    DataInitType, LayerWeightingType, ImageGranularity, ImagePipelineType, ImageNormalizationType, OutputLossType
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline
from model_compression_toolkit.data_generation.common.model_info_exctractors import ActivationExtractor
from model_compression_toolkit.data_generation.common.optimization_utils import AllImagesOptimizationHandler


class DataGenerationConfig:
    """
    Configuration class for data generation.
    """
    def __init__(self,
                 n_iter: int,
                 optimizer: Any,
                 data_gen_batch_size: int,
                 initial_lr: float,
                 output_loss_multiplier: float,
                 image_granularity: ImageGranularity = ImageGranularity.AllImages,
                 scheduler_type: SchedularType = None,
                 bn_alignment_loss_type: BatchNormAlignemntLossType = None,
                 output_loss_type: OutputLossType = None,
                 data_init_type: DataInitType = None,
                 layer_weighting_type: LayerWeightingType = None,
                 image_pipeline_type: ImagePipelineType = None,
                 image_normalization_type: ImageNormalizationType = None,
                 image_padding: int = 0,
                 activations_loss_fn: Callable = None,
                 bn_layer_types: List = [],
                 clip_images: bool = True,
                 reflection: bool = True,
                 ):
        """
        Initialize the DataGenerationConfig.

        Args:
            n_iter (int): Number of iterations for data generation.
            optimizer (Any): The optimizer used for data generation.
            data_gen_batch_size (int): Batch size for data generation.
            initial_lr (float): Initial learning rate for the optimizer.
            output_loss_multiplier (float): Multiplier for the output loss.
            image_granularity (ImageGranularity): Granularity of image data generation. Defaults to ImageGranularity.AllImages.
            scheduler_type (SchedularType): Type of scheduler for the optimizer. Defaults to None.
            bn_alignment_loss_type (BatchNormAlignemntLossType): Type of BatchNorm alignment loss. Defaults to None.
            output_loss_type (OutputLossType): Type of output loss. Defaults to None.
            data_init_type (DataInitType): Type of data initialization. Defaults to None.
            layer_weighting_type (LayerWeightingType): Type of layer weighting. Defaults to None.
            image_pipeline_type (ImagePipelineType): Type of image pipeline. Defaults to None.
            image_normalization_type (ImageNormalizationType): Type of image normalization. Defaults to None.
            image_padding (int): Padding size for images. Defaults to 0.
            activations_loss_fn (Callable): Loss function for activations. Defaults to None.
            bn_layer_types (List): List of BatchNorm layer types. Defaults to [].
            clip_images (bool): Flag to enable image clipping. Defaults to True.
            reflection (bool): Flag to enable reflection. Defaults to True.
        """
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.data_gen_batch_size = data_gen_batch_size
        self.initial_lr = initial_lr
        self.output_loss_multiplier = output_loss_multiplier
        self.image_granularity = image_granularity
        self.scheduler_type = scheduler_type
        self.bn_alignment_loss_type = bn_alignment_loss_type
        self.output_loss_type = output_loss_type
        self.image_pipeline_type = image_pipeline_type
        self.image_normalization_type = image_normalization_type
        self.image_padding = image_padding
        self.data_init_type = data_init_type
        self.layer_weighting_type = layer_weighting_type
        self.activations_loss_fn = activations_loss_fn
        self.bn_layer_types = bn_layer_types
        self.clip_images = clip_images
        self.reflection = reflection

    def get_all_images_optimization_handler(self,
                                            model: Any,
                                            init_dataset: Any,
                                            image_pipeline: BaseImagePipeline,
                                            activation_extractor: ActivationExtractor,
                                            scheduler_step_fn: Callable,
                                            scheduler: Any) -> AllImagesOptimizationHandler:
        """
        Get the AllImagesOptimizationHandler object.

        Args:
            model (Any): Model for which data generation is being performed.
            init_dataset (Any): Initial dataset for data generation.
            image_pipeline (BaseImagePipeline): Callable image pipeline for image manipulation.
            activation_extractor (ActivationExtractor): The activation extractor for the model.
            scheduler_step_fn (Callable): Scheduler step function for the optimizer.
            scheduler (Any): Scheduler for the optimizer.

        Returns:
            AllImagesOptimizationHandler: AllImagesOptimizationHandler object.
        """
        raise NotImplemented


