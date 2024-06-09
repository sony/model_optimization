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

from typing import Any, Callable

from model_compression_toolkit.xquant.common.dataset_utils import DatasetUtils
import numpy as np

import torch


class PytorchDatasetUtils(DatasetUtils):
    """
    Class with helpful methods for handling different kinds of Pytorch datasets from the user.
    """
    @staticmethod
    def prepare_dataset(dataset: Callable, is_validation: bool, device: str = None):
        """
        Prepare the dataset so calling it will return only inputs for the model (like in the case
        of the representative dataset). For example, when the validation dataset is used, the labels
        should be removed.

        Args:
            dataset: Dataset to prepare.
            is_validation: Whether it's validation dataset or not.
            device: Device to transfer the data to.

        Returns:
            Generator to use for retrieving the dataset inputs.

        """

        def process_data(data: Any, is_validation: bool, device: str):
            """
            Processes individual data samples: Transfer them to the device, convert to torch tensors if needed,
            remove labels if this is a validation dataset.

            Args:
                data: The data sample to process.
                is_validation: A flag indicating if this is a validation dataset.
                device: The device to transfer the data to.

            Returns:
                The data as torch tensors on the desired device.
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

            return data

        for x in dataset():
            yield process_data(x, is_validation, device)
