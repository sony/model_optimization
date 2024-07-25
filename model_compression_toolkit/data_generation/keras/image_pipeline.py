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
from typing import Tuple, Dict, Type, Union, List
import tensorflow as tf

from model_compression_toolkit.data_generation.common.enums import ImagePipelineType
from model_compression_toolkit.data_generation.common.image_pipeline import BaseImagePipeline
from model_compression_toolkit.data_generation.keras.image_operations import Smoothing, random_flip, random_crop, \
    clip_images, create_valid_grid, center_crop


class TensorflowSmoothAugmentationImagePipeline(BaseImagePipeline):
    def __init__(self,
                 output_image_size: Union[int, Tuple[int, int]],
                 extra_pixels: Union[int, Tuple[int, int]],
                 normalization: List[List[int]],
                 image_clipping: bool = False,
                 smoothing_filter_size: int = 3,
                 smoothing_filter_sigma: float = 1.25):
        """
        Initialize the TensorflowCropFlipImagePipeline.

        Args:
            output_image_size (Union[int, Tuple[int, int]]): The output image size.
            extra_pixels (Union[int, Tuple[int, int]]): Extra pixels to add to the input image size. Defaults to 0.
            normalization (List[List[float]]): The image normalization values for processing images during optimization.
            image_clipping (bool): Whether to clip images during optimization.
            smoothing_filter_size (int): The size of the smoothing filter. Defaults to 3.
            smoothing_filter_sigma (float): The standard deviation of the smoothing filter. Defaults to 1.25.
       """
        super(TensorflowSmoothAugmentationImagePipeline, self, ).__init__(output_image_size, extra_pixels, image_clipping, normalization)

        smoothing = Smoothing(smoothing_filter_size, smoothing_filter_sigma)
        # List of image manipulation functions and their arguments.
        self.img_manipulation_list = [(random_flip, {}),
                                      (smoothing, {}),
                                      (random_crop, {'height_crop': self.output_image_size[0],
                                                     'width_crop': self.output_image_size[1]}),
                                      ]

        # List of output image manipulation functions and their arguments.
        self.img_output_finalize_list = [(smoothing, {}),
                                         (center_crop, {'height_crop': self.output_image_size[0],
                                                        'width_crop': self.output_image_size[1]}),
                                         ]
        if image_clipping:
            clip_fn = (clip_images, {'valid_grid': create_valid_grid(self.normalization[0], self.normalization[1])})
            self.img_manipulation_list.append(clip_fn)
            self.img_output_finalize_list.append(clip_fn)

    def get_image_input_size(self) -> Tuple[int, int]:
        """
        Get the size of the input image considering extra pixels.

        Returns:
            Tuple[int, int]: Size of the input image.
        """
        return tuple([o + e for (o, e) in zip(self.output_image_size, self.extra_pixels)])

    def image_input_manipulation(self,
                                 images: tf.Tensor) -> tf.Tensor:
        """
        Apply image manipulation functions to input images.

        Args:
            images (tf.Tensor): Input images.

        Returns:
            tf.Tensor: Manipulated images.
        """

        def manipulate_fn(image):
            for fn, args in self.img_manipulation_list:
                image = fn(image, **args)
            return image

        manipulated_images = manipulate_fn(images)
        return manipulated_images

    def image_output_finalize(self,
                              images: tf.Tensor) -> tf.Tensor:
        """
        Apply image output finalization functions to images.

        Args:
            images (tf.Tensor): Input images.

        Returns:
            tf.Tensor: Finalized images.
        """

        def finalize_fn(image):
            for fn, args in self.img_output_finalize_list:
                image = fn(image, **args)
            return image

        output_finalize_images = finalize_fn(images)
        return output_finalize_images


class TensorflowIdentityImagePipeline(BaseImagePipeline):

    def __init__(self, output_image_size: Union[int, Tuple[int, int]],
                 extra_pixels: Union[int, Tuple[int, int]],
                 normalization: List[List[int]],
                 image_clipping: bool = False
                 ):
        """
        Initialize the TensorflowIdentityImagePipeline.

        Args:
            output_image_size (Union[int, Tuple[int, int]]): The output image size.
            extra_pixels (Union[int, Tuple[int, int]]): Extra pixels to add to the input image size. Defaults to 0.
            normalization (List[List[float]]): The image normalization values for processing images during optimization.
            image_clipping (bool): Whether to clip images during optimization.
        """
        super(TensorflowIdentityImagePipeline, self, ).__init__(output_image_size, extra_pixels, image_clipping, normalization)

    def get_image_input_size(self) -> Tuple[int, int]:
        """
        Get the size of the input image.

        Returns:
            Tuple[int, int]: Size of the input image.
        """
        return self.output_image_size

    def image_input_manipulation(self,
                                 images: tf.Tensor) -> tf.Tensor:
        """
        Apply image manipulation functions to input images.

        Args:
            images (tf.Tensor): Input images.

        Returns:
            tf.Tensor: Manipulated images.
        """
        return images

    def image_output_finalize(self,
                              images: tf.Tensor) -> tf.Tensor:
        """
        Apply image output finalization functions to images.

        Args:
            images (tf.Tensor): Input images.

        Returns:
            tf.Tensor: Finalized images.
        """
        return images


# Dictionary mapping ImagePipelineType to corresponding image pipeline classes
image_pipeline_dict: Dict[ImagePipelineType, Type[BaseImagePipeline]] = {
    ImagePipelineType.IDENTITY: TensorflowIdentityImagePipeline,
    ImagePipelineType.SMOOTHING_AND_AUGMENTATION: TensorflowSmoothAugmentationImagePipeline
}
