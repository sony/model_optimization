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

import model_compression_toolkit as mct
from torchvision.models import mobilenet_v2
from PIL import Image
from torchvision import transforms

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


def np_to_pil(img):
    return Image.fromarray(img)


if __name__ == '__main__':

    # Set the batch size of the images at each calibration iteration.
    batch_size = 50

    # Set the path to the folder of images to load and use for the representative dataset.
    # Notice that the folder have to contain at least one image.
    folder = 'path/to/images/folder'

    # Create a representative data generator, which returns a list of images.
    # The images can be preprocessed using a list of preprocessing functions.
    from model_compression_toolkit import FolderImageLoader, MixedPrecisionQuantizationConfig

    image_data_loader = FolderImageLoader(folder,
                                          preprocessing=[np_to_pil,
                                                         transforms.Compose([
                                                             transforms.Resize(256),
                                                             transforms.CenterCrop(224),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                  std=[0.229, 0.224, 0.225]),
                                                         ])
                                                         ],
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
    model = mobilenet_v2()

    # Set the number of calibration iterations to 10.
    num_iter = 10

    # Create a mixed-precision quantization configuration with possible mixed-precision search options.
    # MCTwill search a mixed-precision configuration (namely, bit-width for each layer)
    # and quantize the model according to this configuration.
    # The candidates bit-width for quantization should be defined in the target platform model:
    configuration = MixedPrecisionQuantizationConfig()

    # Get a TargetPlatformCapabilities object that models the hardware for the quantized model inference.
    # In this example, we use a pre-defined platform that allows us to set a non-uniform (LUT) quantizer
    # for low precision weights candidates.
    # The used platform is attached to a Pytorch layers representation.
    target_platform_cap = mct.get_target_platform_capabilities('pytorch', 'default', 'v3_lut')

    # Get KPI information to constraint your model's memory size.
    # Retrieve a KPI object with helpful information of each KPI metric,
    # to constraint the quantized model to the desired memory size.
    kpi_data = mct.pytorch_kpi_data(model,
                                    representative_data_gen,
                                    configuration,
                                    target_platform_capabilities=target_platform_cap)

    # Set a constraint for each of the KPI metrics.
    # Create a KPI object to limit our returned model's size. Note that this values affects only layers and attributes
    # that should be quantized (for example, the kernel of Conv2D in Pytorch will be affected by this value,
    # while the bias will not):
    kpi = mct.KPI(kpi_data.weights_memory * 0.4)  # About 0.4 of the model's weights memory size when quantized with 8 bits.
    # Note that in this example, activations are quantized with fixed bit-width (non mixed-precision) of 8-bit.

    quantized_model, quantization_info = mct.pytorch_post_training_quantization_mixed_precision(model,
                                                                                                representative_data_gen,
                                                                                                target_kpi=kpi,
                                                                                                n_iter=num_iter,
                                                                                                quant_config=configuration,
                                                                                                target_platform_capabilities=target_platform_cap)
