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
This tutorial demonstrates how a model (more specifically, MobileNetV2) can be
quantized and optimized using the Model Compression Toolkit (MCT).
"""

import argparse

import torch.nn
from torchvision.models import resnet50, ResNet50_Weights
import model_compression_toolkit as mct

from PIL import Image
from torchvision import transforms
import tempfile

def np_to_pil(img):
    return Image.fromarray(img)

def count_model_params(model: torch.nn.Module) -> int:
    # Function to count the total number of parameters in a given Pytorch model.
    return sum(p.numel() for p in model.state_dict().values())

def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representative_dataset_dir", type=str, required=True, default=None,
                        help="folder path for the representative dataset.", )
    parser.add_argument("--batch_size", type=int, default=50, help="batch size for the representative data.", )
    parser.add_argument("--num_calibration_iterations", type=int, default=10,
                        help="number of iterations for calibration.", )
    parser.add_argument('--num_score_approximations', type=int, default=32,
                        help='Number of scores to estimate the importance of each channel.')
    parser.add_argument('--compression_rate', type=float, help='Compression rate to remove from the dense model.')

    return parser.parse_args()


if __name__ == "__main__":

    # Parse arguments
    args = argument_handler()

    # Create a function to generate representative data used for channels importance approximation.
    image_data_loader = mct.core.FolderImageLoader(args.representative_dataset_dir,
                                                   preprocessing=[np_to_pil,
                                                                  transforms.Compose(
                                                                      [transforms.Resize(256),
                                                                       transforms.CenterCrop(224),
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std=[0.229, 0.224, 0.225])])],
                                                   batch_size=args.batch_size,
                                                   )


    def representative_data_gen() -> list:
        for _ in range(args.num_calibration_iterations):
            yield [image_data_loader.sample()]


    # Retrieve the target platform capabilities which include the SIMD size configuration for each layer.
    target_platform_cap = mct.get_target_platform_capabilities("pytorch", "default")

    # Load a dense ResNet50 model for pruning. Compute the number of params to
    # initialize the KPI to constraint the memory footprint of the pruned model's weights.
    dense_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1.DEFAULT)
    dense_nparams = count_model_params(dense_model)
    print(f"The model has {dense_nparams} parameters.")

    # Define KPI for pruning. Each float32 parameter requires 4 bytes,
    # hence we multiply the total parameter count by 4 to calculate the memory footprint.
    kpi = mct.KPI(weights_memory=dense_nparams * 4 * args.compression_rate)

    # Create PruningConfig with the number of approximations MCT will compute as importance metric
    # for each channel when using LFH metric to set scores for each output channel that can be removed.
    pruning_config = mct.pruning.PruningConfig(num_score_approximations=args.num_score_approximations)

    # Prune the model.
    pruned_model, pruning_info = mct.pruning.pytorch_pruning_experimental(model=dense_model,
                                                                        target_kpi=kpi,
                                                                        representative_data_gen=representative_data_gen,
                                                                        target_platform_capabilities=target_platform_cap,
                                                                        pruning_config=pruning_config)

    # Count number of params in the pruned model and save it.
    pruned_nparams = count_model_params(pruned_model)
    print(f"The pruned model has {pruned_nparams} parameters.")

    # Export quantized model to ONNX
    _, onnx_file_path = tempfile.mkstemp('.onnx')  # Path of exported model
    mct.exporter.pytorch_export_model(model=pruned_model,
                                      save_model_path=onnx_file_path,
                                      repr_dataset=representative_data_gen)
