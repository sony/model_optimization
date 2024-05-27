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
from typing import Any, Tuple, Union

from model_compression_toolkit.logger import Logger


class BaseImagePipeline(ABC):
    def __init__(self,
                 output_image_size: Union[int, Tuple[int, int]],
                 extra_pixels: Union[int, Tuple[int, int]] = 0):
        """
        Base class for image pipeline.

        Args:
            output_image_size (Union[int, Tuple[int, int]]): The desired output image size.
            extra_pixels Union[int, Tuple[int, int]]: Extra pixels to add to the input image size. Defaults to 0.
        """
        if isinstance(output_image_size, int):
            self.output_image_size = (output_image_size, output_image_size)
        elif isinstance(output_image_size, tuple) and len(output_image_size) == 1:
            self.output_image_size = output_image_size + output_image_size # concatenate two tuples
        elif isinstance(output_image_size, tuple) and len(output_image_size) == 2:
            self.output_image_size = output_image_size
        elif isinstance(output_image_size, tuple):
            Logger.critical(f"'output_image_size' should a tuple of length 1 or 2. Got tuple of length {len(output_image_size)}")
        else:
            Logger.critical(f"'output_image_size' should be an int or tuple but type {type(output_image_size)} was received.")

        if isinstance(extra_pixels, int):
            self.extra_pixels = (extra_pixels, extra_pixels)
        elif isinstance(extra_pixels, tuple) and len(extra_pixels) == 1:
            self.extra_pixels = extra_pixels + extra_pixels # concatenate two tuples
        elif isinstance(extra_pixels, tuple) and len(extra_pixels) == 2:
            self.extra_pixels = extra_pixels
        elif isinstance(extra_pixels, tuple):
            Logger.critical(f"'extra_pixels' should a tuple of length 1 or 2. Got tuple of length {len(extra_pixels)}")
        else:
            Logger.critical(f"'extra_pixels' should be an int or tuple but type {type(extra_pixels)} was received.")
    @abstractmethod
    def get_image_input_size(self) -> Tuple:
        """
        Get the size of the input image for the image pipeline.

        Returns:
            int: The input image size.
        """
        raise NotImplemented

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
        raise NotImplemented

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
        raise NotImplemented