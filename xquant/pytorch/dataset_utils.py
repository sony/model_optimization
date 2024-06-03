#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
from typing import Any

from xquant.common.dataset_utils import DatasetUtils
import numpy as np

import torch

class PytorchDatasetUtils(DatasetUtils):

    @staticmethod
    def wrapped_dataset(dataset: Any, is_validation: bool, device: str = None) -> Any:
        """
                Wraps the dataset to ensure it is properly transferred to the device and processed on a given device.

                Args:
                    dataset: The dataset to be wrapped.
                    is_validation: A flag indicating if this is a validation dataset.
                    device: The device to transfer the data to.

                Returns:
                    A generator that yields processed data.
                """

        def process_data(data: Any, is_validation: bool, device: str):
            """
            Processes individual data samples to transfer them to the device.

            Args:
                data: The data sample to process.
                is_validation: A flag indicating if this is a validation dataset.
                device: The device to transfer the data to.

            Returns:
                A generator that yields the processed data.
            """

            def transfer_to_device(_data):
                if isinstance(_data, np.ndarray):
                    return torch.from_numpy(_data).to(device)
                return _data.to(device)

            if is_validation:
                inputs = data[0]  # Assume data[0] contains the inputs and data[1] the labels
                if isinstance(inputs, list):
                    data = [transfer_to_device(t) for t in inputs]
                else:
                    data = [transfer_to_device(inputs)]
            else:
                data = [transfer_to_device(t) for t in data]

            yield data

        for x in dataset():
            return process_data(x, is_validation, device)

