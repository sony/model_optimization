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
from enum import Enum
from typing import Callable, Any, List


class ImageGranularity(Enum):
    """
    An enum for choosing the image dependence granularity when generating images.
    0. ImageWise
    1. BatchWise
    2. AllImages
    """

    ImageWise = 0
    BatchWise = 1
    AllImages = 2

    @classmethod
    def get_values(cls):
        """
        Get the list of values corresponding to the enum members.

        Returns:
            List of values.
        """
        return [x.value for x_name, x in cls.__members__.items()]


class BaseImagePipeline:
    def __init__(self, output_image_size: int, padding: int = 0):
        """
        Base class for image pipeline.

        Args:
            output_image_size: The desired output image size.
            padding: Padding size for the image.
        """
        self.output_image_size = output_image_size
        self.padding = padding

    def get_image_input_size(self):
        """
        Get the size of the input image for the image pipeline.
        """
        raise NotImplemented

    def image_input_manipulation(self, images: Any):
        """
        Perform image input manipulation in the image pipeline.

        Args:
            images: Input images.

        Returns:
            Manipulated images.
        """
        raise NotImplemented

    def image_output_finalize(self, images: Any):
        """
        Perform finalization of output images in the image pipeline.

        Args:
            images: Output images.

        Returns:
            Finalized images.
        """
        raise NotImplemented


class DataGenerationConfig:

    def __init__(self,
                 n_iter: int,
                 optimizer: Any,
                 scheduler: Any,
                 data_gen_batch_size: int,
                 initial_lr: float,
                 image_granularity: ImageGranularity = ImageGranularity.AllImages,
                 scheduler_step_fn: Callable = None,
                 bna_loss_fn: Callable = None,
                 image_pipeline: BaseImagePipeline = None,
                 image_padding: int = 0,
                 image_initialization_fn: Callable = None,
                 layer_weighting_fn: Callable = None,
                 activations_loss_fn: Callable = None,
                 bn_layer_types: List = []
                 ):
        """
        Configuration class for data generation.

        Args:
            n_iter: Number of iterations.
            optimizer: Optimizer used for training.
            scheduler: Learning rate scheduler.
            data_gen_batch_size: Batch size for data generation.
            initial_lr: Initial learning rate.
            image_granularity: Image granularity when optimizing.
            scheduler_step_fn: Function to perform a step for the scheduler.
            bna_loss_fn: Loss function for batch normalization statistics alignment.
            image_pipeline: Image pipeline for image manipulation.
            image_padding: Padding size for the image.
            image_initialization_fn: Function for image initialization.
            layer_weighting_fn: Function for layer weighting.
            activations_loss_fn: Loss function for activations.
            bn_layer_types: List of batch normalization layer types.
        """
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_gen_batch_size = data_gen_batch_size
        self.initial_lr = initial_lr
        self.image_granularity = image_granularity
        self.scheduler_step_fn = scheduler_step_fn
        self.bna_loss_fn = bna_loss_fn
        self.image_pipeline = image_pipeline
        self.image_padding = image_padding
        self.image_initialization_fn = image_initialization_fn
        self.layer_weighting_fn = layer_weighting_fn
        self.activations_loss_fn = activations_loss_fn
        self.bn_layer_types = bn_layer_types

    def get_dimensions_for_average(self) -> list:
        """
        Get the dimensions to be used for averaging based on the image_granularity attribute.

        Returns:
            list: List of dimensions for averaging.
        """
        if self.image_granularity == ImageGranularity.ImageWise:
            return [2, 3]
        else:
            return [0, 2, 3]


