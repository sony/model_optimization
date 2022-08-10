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

"""
This tutorial demonstrates how a model (more specifically, MobileNetV2) can be
quantized and optimized using the Model Compression Toolkit (MCT).
"""

from torchvision.models import mobilenet_v2
import model_compression_toolkit as mct

from PIL import Image
from torchvision import transforms


def np_to_pil(img):
    return Image.fromarray(img)


if __name__ == '__main__':

    # Set the batch size of the images at each calibration iteration.
    batch_size = 50

    # Set the path to the folder of images to load and use for the representative dataset.
    # Notice that the folder have to contain at least one image.
    folder = '/path/to/images/folder'

    # Create a representative data generator, which returns a list of images.
    # The images can be preprocessed using a list of preprocessing functions.
    from model_compression_toolkit import FolderImageLoader

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
    # For example: A model has two input tensors - one with input shape of [3 X 32 X 32] and the second with
    # an input shape of [3 X 224 X 224]. We calibrate the model using batches of 20 images.
    # Calling representative_data_gen() should return a list
    # of two numpy.ndarray objects where the arrays' shapes are [(20, 3, 32, 32), (20, 3, 224, 224)].
    def representative_data_gen() -> list:
        return [image_data_loader.sample()]

    # Get a TargetPlatformModel object that models the hardware for the quantized model inference.
    # The model determines the quantization methods to use during the MCT optimization process.
    # Here, for example, we use the default model that is attached to a Pytorch
    # layers representation.
    target_platform_cap = mct.get_target_platform_capabilities('pytorch', 'default')


    # Create a model and quantize it using the representative_data_gen as the calibration images.
    # Set the number of calibration iterations to 20.
    model = mobilenet_v2(pretrained=True)
    # set quantization configuration
    quantization_config = mct.DEFAULTCONFIG
    # Configure z threshold algorithm for outlier removal. Set z threshold to 16.
    quantization_config.z_threshold = 16
    # run post training quantization on the model to get the quantized model output
    quantized_model, quantization_info = mct.pytorch_post_training_quantization(model,
                                                                                representative_data_gen,
                                                                                target_platform_capabilities=target_platform_cap,
                                                                                n_iter=20)


