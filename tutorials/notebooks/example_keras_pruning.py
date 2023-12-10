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

import argparse
import keras.models

from keras.applications.resnet50 import ResNet50

import model_compression_toolkit as mct
import tempfile
import cv2
import numpy as np




def count_model_params(model):
    return sum([l.count_params() for l in model.layers])

def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('--representative_dataset_dir', type=str, default="/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_train",
                        help='folder path for the representative dataset.')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size for the representative data.')
    parser.add_argument('--num_calibration_iterations', type=int, default=1,
                        help='number of iterations for calibration.')
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
                                                   preprocessing=[],
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
    target_platform_cap = mct.get_target_platform_capabilities('tensorflow',
                                                               'default')

    # Create a model and quantize it using the representative_data_gen as the calibration images.
    # Set the number of calibration iterations.
    dense_model = ResNet50()
    dense_nparams = count_model_params(dense_model)

    kpi = mct.KPI(weights_memory=dense_nparams*4*0.5) # 50% of dense model
    pruned_model, _ = mct.pruning.keras_pruning_experimental(model=dense_model,
                                                             target_kpi=kpi,
                                                             representative_data_gen=representative_data_gen,
                                                             target_platform_capabilities=target_platform_cap,
                                                             pruning_config=mct.pruning.PruningConfig(num_score_approximations=1))

    pruned_nparams = count_model_params(pruned_model)
    cr = pruned_nparams/dense_nparams
    print(f"cr: {cr}")

    _, keras_file_path = tempfile.mkstemp('.keras') # Path of exported model
    print(f"Saving pruned model: {keras_file_path}")
    keras.models.save_model(pruned_model, keras_file_path)

    _, keras_file_path = tempfile.mkstemp('.h5') # Path of exported model
    print(f"Saving pruned model: {keras_file_path}")
    keras.models.save_model(pruned_model, keras_file_path)

