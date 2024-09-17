# ------------------------------------------------------------------------------
# This file contains code from the Ultralytics repository (YOLOv8)
# Copyright (C) 2024  Ultralytics
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

"""
This code is mostly based on Ultralytics implementation. For more details, refer to the original repository:
https://github.com/ultralytics/ultralytics
"""

import numpy as np
from typing import Tuple
import cv2


def yolov8_preprocess(x: np.ndarray, img_mean: float = 0.0, img_std: float = 255.0, pad_values: int = 114,
                      size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess an input image for YOLOv8 model.

    Args:
        x (np.ndarray): Input image as a NumPy array.
        img_mean (float): Mean value used for normalization. Default is 0.0.
        img_std (float): Standard deviation used for normalization. Default is 255.0.
        pad_values (int): Value used for padding. Default is 114.
        size (Tuple[int, int]): Desired output size (height, width). Default is (640, 640).

    Returns:
        np.ndarray: Preprocessed image as a NumPy array.
    """
    h, w = x.shape[:2]  # Image size
    hn, wn = size  # Image new size
    r = max(h / hn, w / wn)
    hr, wr = int(np.round(h / r)), int(np.round(w / r))
    # pad = ((int((hn - hr) / 2), int((hn - hr) / 2 + 0.5)), (int((wn - wr) / 2), int((wn - wr) / 2 + 0.5)), (0, 0))
    pad = (
        (int((hn - hr) / 2), int((hn - hr) / 2 + 0.5)),
        (int((wn - wr) / 2), int((wn - wr) / 2 + 0.5)),
        (0, 0)
    )

    x = np.flip(x, -1) # Flip image channels
    x = cv2.resize(x, (wr, hr), interpolation=cv2.INTER_AREA) # Aspect ratio preserving resize
    x = np.pad(x, pad, constant_values=pad_values) # Padding to the target size
    x = (x - img_mean) / img_std # Normalization
    return x

def yolov8_preprocess_chw_transpose(x: np.ndarray, img_mean: float = 0.0, img_std: float = 255.0, pad_values: int = 114,
                                    size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess an input image for YOLOv8 model with additional CHW transpose (for PyTorch implementation)

    Args:
        x (np.ndarray): Input image as a NumPy array.
        img_mean (float): Mean value used for normalization. Default is 0.0.
        img_std (float): Standard deviation used for normalization. Default is 255.0.
        pad_values (int): Value used for padding. Default is 114.
        size (Tuple[int, int]): Desired output size (height, width). Default is (640, 640).

    Returns:
        np.ndarray: Preprocessed image as a NumPy array.
    """
    return yolov8_preprocess(x, img_mean, img_std, pad_values, size).transpose([2, 0, 1])