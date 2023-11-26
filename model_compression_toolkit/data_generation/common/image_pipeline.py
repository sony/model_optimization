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
from typing import Any


class BaseImagePipeline(ABC):
    def __init__(self,
                 output_image_size: int,
                 extra_pixels: int = 0):
        """
        Base class for image pipeline.

        Args:
            output_image_size (int): The desired output image size.
            extra_pixels (int, optional): Extra pixels to add to the input image size. Defaults to 0.
        """
        self.output_image_size = output_image_size
        self.extra_pixels = extra_pixels
    @abstractmethod
    def get_image_input_size(self) -> int:
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