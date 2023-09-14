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

from model_compression_toolkit.data_generation.common.enums import SchedulerType, BatchNormAlignemntLossType, \
    DataInitType, BNLayerWeightingType, ImageGranularity, ImagePipelineType, ImageNormalizationType, OutputLossType


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
                 scheduler_type: SchedulerType = None,
                 bn_alignment_loss_type: BatchNormAlignemntLossType = None,
                 output_loss_type: OutputLossType = None,
                 data_init_type: DataInitType = None,
                 layer_weighting_type: BNLayerWeightingType = None,
                 image_pipeline_type: ImagePipelineType = None,
                 image_normalization_type: ImageNormalizationType = None,
                 extra_pixels: int = 0,
                 bn_layer_types: List = [],
                 last_layer_types: List = [],
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
            scheduler_type (SchedulerType): Type of scheduler for the optimizer. Defaults to None.
            bn_alignment_loss_type (BatchNormAlignemntLossType): Type of BatchNorm alignment loss. Defaults to None.
            output_loss_type (OutputLossType): Type of output loss. Defaults to None.
            data_init_type (DataInitType): Type of data initialization. Defaults to None.
            layer_weighting_type (BNLayerWeightingType): Type of layer weighting. Defaults to None.
            image_pipeline_type (ImagePipelineType): Type of image pipeline. Defaults to None.
            image_normalization_type (ImageNormalizationType): Type of image normalization. Defaults to None.
            extra_pixels (int): Extra pixels to add to the input image size. Defaults to 0.
            bn_layer_types (List): List of BatchNorm layer types. Defaults to [].
            last_layer_types (List): List of layer types. Defaults to [].
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
        self.extra_pixels = extra_pixels
        self.data_init_type = data_init_type
        self.layer_weighting_type = layer_weighting_type
        self.bn_layer_types = bn_layer_types
        self.last_layer_types = last_layer_types
        self.clip_images = clip_images
        self.reflection = reflection

