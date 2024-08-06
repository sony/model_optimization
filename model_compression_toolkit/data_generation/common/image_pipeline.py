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
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Dict, List

from model_compression_toolkit.data_generation import ImageNormalizationType
from model_compression_toolkit.logger import Logger


class BaseImagePipeline(ABC):
    def __init__(self,
                 output_image_size: Union[int, Tuple[int, int]],
                 extra_pixels: Union[int, Tuple[int, int]] = 0,
                 image_clipping: bool = False,
                 normalization: List[List[int]] = [[0, 0, 0], [1, 1, 1]]):
        """
        Base class for image pipeline.

        Args:
            output_image_size (Union[int, Tuple[int, int]]): The desired output image size.
            extra_pixels (Union[int, Tuple[int, int]]): Extra pixels to add to the input image size. Defaults to 0.
            image_clipping (bool): Whether to clip images during optimization.
            normalization (List[List[float]]): The image normalization values for processing images during optimization.
        """
        if isinstance(output_image_size, int):
            self.output_image_size = (output_image_size, output_image_size)
        elif isinstance(output_image_size, tuple) and len(output_image_size) == 1:
            self.output_image_size = output_image_size + output_image_size # concatenate two tuples
        elif isinstance(output_image_size, tuple) and len(output_image_size) == 2:
            self.output_image_size = output_image_size
        elif isinstance(output_image_size, tuple):
            Logger.critical(f"'output_image_size' should a tuple of length 1 or 2. Got tuple of length {len(output_image_size)}") # pragma: no cover
        else:
            Logger.critical(f"'output_image_size' should be an int or tuple but type {type(output_image_size)} was received.") # pragma: no cover

        if isinstance(extra_pixels, int):
            self.extra_pixels = (extra_pixels, extra_pixels)
        elif isinstance(extra_pixels, tuple) and len(extra_pixels) == 1:
            self.extra_pixels = extra_pixels + extra_pixels # concatenate two tuples
        elif isinstance(extra_pixels, tuple) and len(extra_pixels) == 2:
            self.extra_pixels = extra_pixels
        elif isinstance(extra_pixels, tuple):
            Logger.critical(f"'extra_pixels' should a tuple of length 1 or 2. Got tuple of length {len(extra_pixels)}") # pragma: no cover
        else:
            Logger.critical(f"'extra_pixels' should be an int or tuple but type {type(extra_pixels)} was received.") # pragma: no cover

        self.image_clipping = image_clipping
        self.normalization = normalization

    @abstractmethod
    def get_image_input_size(self) -> Tuple[int, int]:
        """
        Get the size of the input image for the image pipeline.

        Returns:
            Tuple[int, int]: The input image size.
        """
        raise NotImplemented # pragma: no cover

    @abstractmethod
    def image_input_manipulation(self,
                                 images: Any) -> Any:
        """
        Perform image input manipulation in the image pipeline.

        Args:
            images (Any): Input images.

        Returns:
            Any: Manipulated images.
        """
        raise NotImplemented # pragma: no cover

    @abstractmethod
    def image_output_finalize(self,
                              images: Any) -> Any:
        """
        Perform finalization of output images in the image pipeline.

        Args:
            images (Any): Output images.

        Returns:
            Any: Finalized images.
        """
        raise NotImplemented # pragma: no cover


# Dictionary mapping ImageNormalizationType to corresponding normalization values
image_normalization_dict: Dict[ImageNormalizationType, List[List[float]]] = {
    ImageNormalizationType.TORCHVISION: [[0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]],
    ImageNormalizationType.KERAS_APPLICATIONS: [[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]],
    ImageNormalizationType.NO_NORMALIZATION: [[0, 0, 0], [1, 1, 1]]
}