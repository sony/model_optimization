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
import tensorflow as tf

import model_compression_toolkit as mct
import tempfile
import numpy as np
import cv2


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


def count_model_params(model: keras.models.Model) -> int:
    # Function to count the total number of parameters in a given Keras model.
    return sum([l.count_params() for l in model.layers])

def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('--representative_dataset_dir', type=str, help='Folder path for the representative dataset.')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for the representative data.')
    parser.add_argument('--num_score_approximations', type=int, default=32,
                        help='Number of scores to estimate the importance of each channel.')
    parser.add_argument('--compression_rate', type=float, help='Compression rate to remove from the dense model.')

    return parser.parse_args()


if __name__ == '__main__':
    args = argument_handler()

    # Create a function to generate representative data used for channels importance approximation.
    image_data_loader = mct.core.FolderImageLoader(args.representative_dataset_dir,
                                                   preprocessing=[resize,
                                                                  tf.keras.applications.resnet50.preprocess_input],
                                                   batch_size=args.batch_size)

    def representative_data_gen() -> list:
        yield [image_data_loader.sample()]


    # Retrieve the target platform capabilities which include the SIMD size configuration for each layer.
    target_platform_cap = mct.get_target_platform_capabilities('tensorflow',
                                                               'default')

    # Load a dense ResNet50 model for pruning. Compute the number of params to
    # initialize the KPI to constraint the memory footprint of the pruned model's weights.
    dense_model = ResNet50()
    dense_nparams = count_model_params(dense_model)
    print(f"Model has {dense_nparams} parameters.")
    kpi = mct.KPI(weights_memory=dense_nparams * 4 * args.compression_rate)

    # Create PruningConfig with the number of approximations MCT will compute as importance metric
    # for each channel when using LFH metric to set scores for each output channel that can be removed.
    pruning_config = mct.pruning.PruningConfig(num_score_approximations=args.num_score_approximations)

    # Prune the model.
    pruned_model, pruning_info = mct.pruning.keras_pruning_experimental(model=dense_model,
                                                                        target_kpi=kpi,
                                                                        representative_data_gen=representative_data_gen,
                                                                        target_platform_capabilities=target_platform_cap,
                                                                        pruning_config=pruning_config)

    # Count number of params in the pruned model and save it.
    pruned_nparams = count_model_params(pruned_model)
    print(f"Pruned model has {pruned_nparams} parameters.")
    _, keras_file_path = tempfile.mkstemp('.keras')
    print(f"Saving pruned model: {keras_file_path}")
    keras.models.save_model(pruned_model, keras_file_path)

