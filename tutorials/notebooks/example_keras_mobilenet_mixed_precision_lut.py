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

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import model_compression_toolkit as mct
import tempfile

"""
Mixed precision is a method for quantizing a model using different bit widths
for different layers of the model. 
This tutorial demonstrates how to use mixed-precision in MCT to
quantize MobileNetV2 weights, using non-uniform,
lookup table-based quantizer for low precision quantization (2 and 4 bits)
MCT supports non-uniform mixed-precision for weights quantization only.
In this example, activations are quantized with fixed 8-bit precision. 
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
    parser.add_argument('--weights_compression_ratio', type=float, default=0.4,
                        help='weights compression ratio.')
    parser.add_argument('--mixed_precision_num_of_images', type=int, default=32,
                        help='number of images to use for mixed-precision configuration search.')
    parser.add_argument('--enable_mixed_precision_gradients_weighting', action='store_true', default=False,
                        help='Whether to use gradients during mixed-precision configuration search or not.')

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
    image_data_loader = mct.core.FolderImageLoader(folder,
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
        for _ in range(args.num_calibration_iterations):
            yield [image_data_loader.sample()]

    # Create a model to quantize.
    model = MobileNetV2()

    # Create a mixed-precision quantization configuration with possible mixed-precision search options.
    # MCT will search a mixed-precision configuration (namely, bit-width for each layer)
    # and quantize the model according to this configuration.
    # The candidates bit-width for quantization should be defined in the target platform model:
    configuration = mct.core.CoreConfig(mixed_precision_config=mct.core.MixedPrecisionQuantizationConfigV2(num_of_images=args.mixed_precision_num_of_images,
                                                                                                           use_grad_based_weights=args.enable_mixed_precision_gradients_weighting))

    # Get a TargetPlatformCapabilities object that models the hardware for the quantized model inference.
    # In this example, we use a pre-defined platform that allows us to set a non-uniform (LUT) quantizer
    # for low precision weights candidates.
    # The used platform is attached to a Tensorflow layers representation.
    target_platform_cap = mct.get_target_platform_capabilities('tensorflow', 'default', 'v3_lut')

    # Get KPI information to constraint your model's memory size.
    # Retrieve a KPI object with helpful information of each KPI metric,
    # to constraint the quantized model to the desired memory size.
    kpi_data = mct.core.keras_kpi_data_experimental(model,
                                                    representative_data_gen,
                                                    configuration,
                                                    target_platform_capabilities=target_platform_cap)

    # Set a constraint for each of the KPI metrics.
    # Create a KPI object to limit our returned model's size. Note that this values affects only layers and attributes
    # that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value,
    # while the bias will not):
    # examples:
    # weights_compression_ratio = 0.4 - About 0.4 of the model's weights memory size when quantized with 8 bits.
    kpi = mct.core.KPI(kpi_data.weights_memory * args.weights_compression_ratio)
    # Note that in this example, activations are quantized with fixed bit-width (non mixed-precision) of 8-bit.

    quantized_model, quantization_info = mct.ptq.keras_post_training_quantization_experimental(model,
                                                                                               representative_data_gen,
                                                                                               target_kpi=kpi,
                                                                                               core_config=configuration,
                                                                                               target_platform_capabilities=target_platform_cap)

    # Export quantized model to TFLite and Keras.
    # For more details please see: https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/exporter/README.md
    _, tflite_file_path = tempfile.mkstemp('.tflite') # Path of exported model
    mct.exporter.keras_export_model(model=quantized_model, save_model_path=tflite_file_path,
                                    target_platform_capabilities=target_platform_cap,
                                    serialization_format=mct.exporter.KerasExportSerializationFormat.TFLITE)

    _, keras_file_path = tempfile.mkstemp('.h5') # Path of exported model
    mct.exporter.keras_export_model(model=quantized_model, save_model_path=keras_file_path,
                                    target_platform_capabilities=target_platform_cap,
                                    serialization_format=mct.exporter.KerasExportSerializationFormat.KERAS_H5)