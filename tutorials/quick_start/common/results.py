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
import csv
import logging
from os import path
from typing import List, Union

from model_compression_toolkit.core.common.user_info import UserInformation
from common.constants import MODEL_NAME, MODEL_LIBRARY, VALIDATION_DATASET_FOLDER

from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities


class DatasetInfo:
    """
    Holds information about the evaluation dataset.
    """
    def __init__(self, dataset_name: str, n_images: int):
        """
        Initializes a new instance of the DatasetInfo class.

        Args:
            dataset_name (str): The name of the dataset.
            n_images (int): The number of images in the dataset.
        """
        self.dataset_name = dataset_name
        self.n_images = n_images


class QuantInfo:
    """
    Holds information about the quantization process.
    """
    def __init__(self, user_info: UserInformation,
                 tpc_info: dict,
                 technique: str
                 ):
        """
        Initializes a new instance of the QuantInfo class.

        Args:
            user_info (UserInformation): Quantization information returned from MCT
            tpc_info (dict): The target platform capabilities information which is provided to the MCT.
            technique (str): String to describe the quantization method.
        """
        self.user_info = user_info
        self.tpc_info = tpc_info
        self.technique = technique


def read_models_list(filename: str) -> csv.DictReader:
    """
    Reads a models list and parameters from a CSV file and returns csv.DictReader.

    Args:
        filename: Path to the CSV file containing the models list.

    Returns:
        A dictionary read from the CSV file.

    """
    return csv.DictReader(open(path.join(filename)))


def write_results(filename: str, models_list: List[dict], fieldnames: List[str]):
    """
    Writes the results of a models list to a CSV file.

    Args:
        filename: Path to the CSV file to write the results.
        models_list: List of model results, where each result is represented by a dictionary.
        fieldnames: List of fieldnames to be written as headers in the CSV file.

    Returns:
        None

    """
    writer = csv.DictWriter(open(filename, 'w'), fieldnames=fieldnames)
    writer.writeheader()
    for item in models_list:
        writer.writerow(item)


def parse_results(params: dict, float_acc: float, quant_acc: float, quant_info: QuantInfo, dataset_info: DatasetInfo) -> dict:
    """
    Parses the results of model evaluation and quantization into a dictionary format.

    Args:
        params: Dictionary of parameters containing information about the model, dataset, etc.
        float_acc: Floating-point accuracy of the model.
        quant_acc: Quantized accuracy of the model.
        quant_info: Quantization information object containing details of the quantized model.

    Returns:
        A dictionary containing the parsed results.

    """
    a_bits = quant_info.tpc_info['Target Platform Model']['Default quantization config']['activation_n_bits']
    w_bits = quant_info.tpc_info['Target Platform Model']['Default quantization config']['weights_n_bits']
    bit_config = 'W' + str(w_bits) + 'A' + str(a_bits)

    res = {}
    res['ModelName'] = params[MODEL_NAME]
    res['ModelLibrary'] = params[MODEL_LIBRARY]
    res['DatasetName'] = dataset_info.dataset_name
    res['TotalImages'] = dataset_info.n_images
    res['FloatAcc'] = round(float_acc, 4)
    res['QuantAcc'] = round(quant_acc, 4)
    res['Size[MB]'] = round(quant_info.user_info.final_kpi.weights_memory / 1e6, 2)
    res['BitsConfig'] = bit_config
    res['QuantTechnique'] = quant_info.technique
    res['TPC'] = quant_info.tpc_info['Target Platform Capabilities'] + '-' + quant_info.tpc_info['Version']

    return res


def plot_results(data: Union[dict, List[dict]], spacing=15):
    """
    Prints the given dictionary or list of dictionaries as a table to the screen.

    Args:
        data (Union[dict, List[dict]]): The dictionary or list of dictionaries to be printed as a table.
    """
    # Handle a single dictionary
    if isinstance(data, dict):
        data = [data]

    # Extract the keys as the table header
    header = list(data[0].keys())

    # Print the header
    header_title = [h + (spacing - len(h)) * " " for h in header]
    logging.info("\t".join(header_title))

    # Print the rows
    for item in data:
        values = [str(item.get(key, "")) for key in header]
        values = [v + (spacing - len(v)) * " " for v in values]
        logging.info("\t".join(values))
        
        
