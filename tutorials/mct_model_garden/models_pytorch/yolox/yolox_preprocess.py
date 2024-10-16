# ===========================================================================================
# The following code was adopted from https://github.com/Megvii-BaseDetection/YOLOX
# ===========================================================================================

import numpy as np
from typing import Tuple
import cv2


def yolox_preprocess_chw_transpose(img: np.ndarray,
                                   pad_values: int = 114,
                                   size: Tuple[int, int] = (416, 416)) -> np.ndarray:
    """
    Preprocess an input image for YOLOX model with reshape and CHW transpose (for PyTorch implementation)

    Args:
        img (np.ndarray): Input image as a NumPy array.
        pad_values (int): Value used for padding. Default is 114.
        size (Tuple[int, int]): Desired output size (height, width). Default is (416, 416).

    Returns:
        np.ndarray: Preprocessed image as a NumPy array.
    """
    padded_img = np.ones((size[0], size[1], 3), dtype=np.uint8) * pad_values
    r = min(size[0] / img.shape[0], size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose((2, 0, 1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img
