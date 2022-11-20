# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

import argparse
from keras.applications.mobilenet_v2 import MobileNetV2

import model_compression_toolkit as mct

"""
This tutorial demonstrates how a model (more specifically, MobileNetV2) can be
quantized and optimized using the Model Compression Toolkit (MCT) with 
mixed-precision quantization and GPTQ (gradient-based PTQ). 
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


def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('--representative_dataset_dir', type=str, required=True, default=None,
                        help='folder path for the representative dataset.')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size for the representative data.')
    parser.add_argument('--num_calibration_iterations', type=int, default=10,
                        help='number of iterations for calibration.')
    parser.add_argument('--num_gptq_training_iterations', type=int, default=5000,
                        help='number of iterations for gptq training.')
    parser.add_argument('--weights_compression_ratio', type=float, default=0.75,
                        help='weights compression ratio for model memory size reduction.')
    return parser.parse_args()


if __name__ == '__main__':

    # Parse arguments
    args = argument_handler()

    # Set the batch size of the images at each calibration iteration.
    batch_size = args.batch_size

    # Set the path to the folder of images to load and use for the representative dataset.
    # Notice that the folder have to contain at least one image.
    folder = args.representative_dataset_dir

    # Create a representative data generator, which returns a list of images.
    # The images can be preprocessed using a list of preprocessing functions.
    from model_compression_toolkit import FolderImageLoader

    image_data_loader = FolderImageLoader(folder,
                                          preprocessing=[resize, normalization],
                                          batch_size=batch_size)


    # Create a Callable representative dataset for calibration purposes.
    # The function should be called without any arguments, and should return a list numpy arrays (array for each
    # model's input).
    # For example: A model has two input tensors - one with input shape of [32 X 32 X 3] and the second with
    # an input shape of [224 X 224 X 3]. We calibrate the model using batches of 20 images.
    # Calling representative_data_gen() should return a list
    # of two numpy.ndarray objects where the arrays' shapes are [(20, 3, 32, 32), (20, 3, 224, 224)].
    def representative_data_gen() -> list:
        for _ in range(args.num_calibration_iterations):
            yield [image_data_loader.sample()]


    # Get a TargetPlatformModel object that models the hardware for the quantized model inference.
    # The model determines the quantization methods to use during the MCT optimization process.
    # Here, for example, we use the default target platform model that is attached to a Tensorflow
    # layers representation.
    target_platform_cap = mct.get_target_platform_capabilities('tensorflow', 'default')

    # Create a model
    model = MobileNetV2()

    # Create a mixed-precision quantization configuration.
    mixed_precision_config = mct.MixedPrecisionQuantizationConfigV2()

    # Create a core quantization configuration, set the mixed-precision configuration,
    # and set the number of calibration iterations.
    config = mct.CoreConfig(mixed_precision_config=mixed_precision_config)

    # Get KPI information to constraint your model's memory size.
    # Retrieve a KPI object with helpful information of each KPI metric,
    # to constraint the quantized model to the desired memory size.
    kpi_data = mct.keras_kpi_data_experimental(model,
                                               representative_data_gen,
                                               config,
                                               target_platform_capabilities=target_platform_cap)

    # Set a constraint for each of the KPI metrics.
    # Create a KPI object to limit our returned model's size. Note that this values affects only layers and attributes
    # that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value,
    # while the bias will not)
    # examples:
    # weights_compression_ratio = 0.75 - About 0.75 of the model's weights memory size when quantized with 8 bits.
    kpi = mct.KPI(kpi_data.weights_memory * args.weights_compression_ratio)

    # Create a GPTQ quantization configuration and set the number of training iterations.
    gptq_config = mct.get_keras_gptq_config(n_iter=args.num_gptq_training_iterations)

    quantized_model, quantization_info = mct.keras_gradient_post_training_quantization_experimental(model,
                                                                                                    representative_data_gen,
                                                                                                    gptq_config=gptq_config,
                                                                                                    core_config=config,
                                                                                                    target_platform_capabilities=target_platform_cap,
                                                                                                    target_kpi=kpi)
