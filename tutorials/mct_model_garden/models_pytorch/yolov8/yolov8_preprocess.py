# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
"""
This code is mostly based on Ultralytics implementation. For more details, refer to the original repository:
https://github.com/ultralytics/ultralytics
"""

import numpy as np
from typing import Tuple
import cv2


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

    h, w = x.shape[:2]  # Image size
    hn, wn = size  # Image new size
    r = max(h / hn, w / wn)
    hr, wr = int(np.round(h / r)), int(np.round(w / r))
    pad = (
        (int((hn - hr) / 2), int((hn - hr) / 2 + 0.5)),
        (int((wn - wr) / 2), int((wn - wr) / 2 + 0.5)),
        (0, 0)
    )

    x = np.flip(x, -1)  # Flip image channels
    x = cv2.resize(x, (wr, hr), interpolation=cv2.INTER_AREA)  # Aspect ratio preserving resize
    x = np.pad(x, pad, constant_values=pad_values)  # Padding to the target size
    x = (x - img_mean) / img_std  # Normalization
    x = x.transpose([2, 0, 1])
    return x
