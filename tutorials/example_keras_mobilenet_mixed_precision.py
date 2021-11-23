# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

import model_compression_toolkit as mct
from tensorflow.keras.applications.mobilenet import MobileNet

"""
Mixed precision is a method for quantizing a model using different bit widths
for different layers of the model. 
This tutorial demonstrates how to use mixed-precision in MCT to
quantize MobileNetV1.
For now, MCT supports mixed-precision for weights only. 
"""

####################################
# Preprocessing images
####################################
import cv2
import numpy as np

MEAN = 127.5
STD = 127.5
RESIZE_SCALE = 256 / 224
SIZE = 224


def resize(x):
    resize_side = max(RESIZE_SCALE * SIZE / x.shape[0], RESIZE_SCALE * SIZE / x.shape[1])
    height_tag = int(np.round(resize_side * x.shape[0]))
    width_tag = int(np.round(resize_side * x.shape[1]))
    resized_img = cv2.resize(x, (width_tag, height_tag))
    offset_height = int((height_tag - SIZE) / 2)
    offset_width = int((width_tag - SIZE) / 2)
    cropped_img = resized_img[offset_height:offset_height + SIZE, offset_width:offset_width + SIZE]
    return cropped_img


def normalization(x):
    return (x - MEAN) / STD


if __name__ == '__main__':

    # Set the batch size of the images at each calibration iteration.
    batch_size = 50

    # Set the path to the folder of images to load and use for the representative dataset.
    # Notice that the folder have to contain at least one image.
    folder = '/path/to/images/folder'

    # Create a representative data generator, which returns a list of images.
    # The images can be preprocessed using a list of preprocessing functions.
    from model_compression_toolkit import FolderImageLoader, MixedPrecisionQuantizationConfig

    image_data_loader = FolderImageLoader(folder,
                                          preprocessing=[resize, normalization],
                                          batch_size=batch_size)

    # Create a Callable representative dataset for calibration purposes.
    # The function should be called without any arguments, and should return a list numpy arrays (array for each
    # model's input).
    # For example: if the model has two input tensors - one with input shape of 32X32X3 and the second with input
    # shape of 224X224X3, and we calibrate the model using batches of 20 images,
    # calling representative_data_gen() should return a list
    # of two numpy.ndarray objects where the arrays' shapes are [(20, 32, 32, 3), (20, 224, 224, 3)].
    def representative_data_gen() -> list:
        return [image_data_loader.sample()]

    # Create a model to quantize.
    model = MobileNet()

    # Set the number of calibration iterations to 10.
    num_iter = 10

    # Create a mixed-precision configuration with possible bit widths. MCT
    # will search a mixed-precision configuration (namely, bit width for each layer)
    # and quantize the model according to this configuration.
    # Here, each layer can be quantized by 2, 4 or 8 bits.
    configuration = MixedPrecisionQuantizationConfig(weights_n_bits=[2, 8, 4])

    # Create a KPI object to limit our returned model's size. Note that this value affects only coefficients that
    # should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value, while the bias
    # will not):
    kpi = mct.KPI(model.count_params() * 0.75)  # About 0.75 of the model size when quantized with 8 bits.

    quantized_model, quantization_info = mct.keras_post_training_quantization_mixed_precision(model,
                                                                                              representative_data_gen,
                                                                                              n_iter=num_iter,
                                                                                              quant_config=configuration,
                                                                                              target_kpi=kpi)
