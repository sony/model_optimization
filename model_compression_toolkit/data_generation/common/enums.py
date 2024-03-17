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


class EnumBaseClass(Enum):
    @classmethod
    def get_values(cls):
        """
        Get the list of values corresponding to the enum members.

        Returns:
            List of values.
        """
        return [value.value for value in cls.__members__.values()]


class ImageGranularity(EnumBaseClass):
    """
    An enum for choosing the image dependence granularity when generating images:

    ImageWise

    BatchWise

    AllImages

    """

    ImageWise = 0
    BatchWise = 1
    AllImages = 2


class DataInitType(EnumBaseClass):
    """
    An enum for choosing the image dependence granularity when generating images:

    Gaussian

    Diverse

    """
    Gaussian = 0
    Diverse = 1


class ImagePipelineType(EnumBaseClass):
    """
    An enum for choosing the image pipeline type for image manipulation:

    RANDOM_CROP - Crop the images.

    RANDOM_CROP_FLIP - Crop and flip the images.

    IDENTITY - Do not apply any manipulation (identity transformation).

    """
    RANDOM_CROP = 'random_crop'
    RANDOM_CROP_FLIP = 'random_crop_flip'
    IDENTITY = 'identity'


class ImageNormalizationType(EnumBaseClass):
    """
    An enum for choosing the image normalization type:

    TORCHVISION - Normalize the images using torchvision normalization.

    KERAS_APPLICATIONS - Normalize the images using keras_applications imagenet normalization.

    NO_NORMALIZATION - Do not apply any normalization.

    """
    TORCHVISION = 'torchvision'
    KERAS_APPLICATIONS = 'keras_applications'
    NO_NORMALIZATION = 'no_normalization'


class BNLayerWeightingType(EnumBaseClass):
    """
   An enum for choosing the layer weighting type:

   AVERAGE - Use the same weight per layer.

   FIRST_LAYER_MULTIPLIER - Use a multiplier for the first layer, all other layers with the same weight.

   GRAD - Use gradient-based layer weighting.

   """
    AVERAGE = 'average'
    FIRST_LAYER_MULTIPLIER = 'first_layer_multiplier'
    GRAD = 'grad'


class BatchNormAlignemntLossType(EnumBaseClass):
    """
    An enum for choosing the BatchNorm alignment loss type:

    L2_SQUARE - Use L2 square loss for BatchNorm alignment.

    """
    L2_SQUARE = 'l2_square'


class OutputLossType(EnumBaseClass):
    """
    An enum for choosing the output loss type:

    NONE - No output loss is applied.

    MIN_MAX_DIFF - Use min-max difference as the output loss.

    REGULARIZED_MIN_MAX_DIFF - Use regularized min-max difference as the output loss.

    """
    NONE = 'none'
    MIN_MAX_DIFF = 'min_max_diff'
    REGULARIZED_MIN_MAX_DIFF = 'regularized_min_max_diff'


class SchedulerType(EnumBaseClass):
    """
    An enum for choosing the scheduler type for the optimizer:

    REDUCE_ON_PLATEAU - Use the ReduceOnPlateau scheduler.

    STEP - Use the Step scheduler.

    """
    REDUCE_ON_PLATEAU = 'reduce_on_plateau'
    STEP = 'step'
