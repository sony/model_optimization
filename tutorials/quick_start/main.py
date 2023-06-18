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
import importlib
import logging
from typing import Dict, Tuple

from common.results import write_results, read_models_list, parse_results, QuantInfo, plot_results
from common.utils import find_modules
from common.constants import MODEL_NAME, MODEL_LIBRARY, OUTPUT_RESULTS_FILE, TARGET_PLATFORM_NAME, \
    TARGET_PLATFORM_VERSION


# Script to Evaluate and Compress Pre-trained Neural Network Model(s) using MCT (Model Compression Toolkit)

def argument_handler():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', '-m', type=str, required=True,
                        help='The name of the pre-trained model to run')
    parser.add_argument('--model_library', type=str, required=True,
                        help='The library that contains the pre-trained model')
    parser.add_argument('--validation_dataset_folder', type=str, required=True,
                        help='Path to the validation dataset')
    parser.add_argument('--representative_dataset_folder', type=str, required=True,
                        help='Path to the representative dataset used for quantization')
    parser.add_argument('--num_representative_images', type=int, default=200,
                        help='Number of images for representative dataset')
    parser.add_argument('--target_platform_name', type=str, default='default',
                        help='Specifies the name of the target platform capabilities (TPC) to select from the available TPCs provided by MCT')
    parser.add_argument('--target_platform_version', type=str, default='latest',
                        help='Specifies the version of the target platform capabilities (TPC) to select from the available TPCs provided by MCT')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for accuracy evaluation')
    parser.add_argument('--models_list_csv', type=str, default=None,
                        help='Run according to a list of models and parameters taken from a csv file')
    parser.add_argument('--output_results_file', type=str, default='model_quantization_results.csv',
                        help='Run according to a list of models and parameters taken from a csv file')
    parser.add_argument('--validation_set_limit', type=int, default=None,
                        help='Limits the number of images taken for evaluation')
    parser.add_argument('--export_model', action="store_true",
                        help='Whether to export the model after quantization')

    args = parser.parse_args()
    return args


def quantization_flow(config: Dict) -> Tuple[float, float, QuantInfo]:
    """
    This function implements the typical workflow when using MCT.
    It begins by evaluating the performance of the floating-point model.
    Next, the model is compressed using MCT quantization techniques.
    Finally, the evaluation process is repeated on the compressed model to assess its performance.

    Args:
        config (Dict): Configurations dictionary that contains the settings for the quantization flow

    Returns:
        float_results (float): The accuracy of the floating point model
        quant_results (float): The accuracy of the quantized model
        quantization_info (QuantInfo): Information of the model optimization process from MCT
    """

    # Find and import the required modules for the models collection library ("model_library")
    model_lib_module, quant_module = find_modules(config[MODEL_LIBRARY])
    model_lib = importlib.import_module(model_lib_module)
    quant = importlib.import_module(quant_module)

    # Create a ModelLibrary object that combines the selected pre-trained model with the corresponding performance
    # evaluation functionality
    ml = model_lib.ModelLib(config)

    # Get the floating point model
    float_model = ml.get_model()

    # Evaluate the float model
    float_results, _ = ml.evaluate(float_model)

    # Run model compression toolkit
    target_platform_cap = quant.get_tpc(config[TARGET_PLATFORM_NAME], config[TARGET_PLATFORM_VERSION])
    quantized_model, quantization_info = quant.quantize(float_model,
                                                        ml.get_representative_dataset,
                                                        target_platform_cap,
                                                        config)

    # Evaluate quantized model
    quant_results, dataset_info = ml.evaluate(quantized_model)


    return float_results, quant_results, quantization_info, dataset_info


if __name__ == '__main__':

    # Set arguments and parameters
    args = argument_handler()

    # Set logger level
    logging.getLogger().setLevel(logging.INFO)

    # Create configuration dictionary
    config = dict(args._get_kwargs())

    if args.models_list_csv is None:

        # Run quantization flow and add results to the table
        float_acc, quant_acc, quant_info, dataset_info = quantization_flow(config)

        # Parse the results and print to screen
        res = parse_results(config, float_acc, quant_acc, quant_info, dataset_info)
        plot_results(res)

    else:
        models_list = read_models_list(args.models_list_csv)
        results_table = []
        for p in models_list:

            # Get next model and parameters from the list
            logging.info(f"Testing model: {p[MODEL_NAME]} from library: {p[MODEL_LIBRARY]}")
            config.update(p)

            # Run quantization flow and add results to the table
            float_acc, quant_acc, quant_info, dataset_info = quantization_flow(config)

            # Add results to the table
            res = parse_results(config, float_acc, quant_acc, quant_info, dataset_info)
            results_table.append(res)

        # Store results table
        write_results(config[OUTPUT_RESULTS_FILE], results_table, res.keys())

        # Print results to screen
        plot_results(results_table)

