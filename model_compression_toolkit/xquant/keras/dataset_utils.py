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

from model_compression_toolkit.xquant.common.dataset_utils import DatasetUtils


class KerasDatasetUtils(DatasetUtils):
    """
    Class with helpful methods for handling different kinds of Keras datasets from the user.
    """

    @staticmethod
    def prepare_dataset(dataset: Any, is_validation: bool, device: str = None):
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
        def process_data(x: Any, is_validation: bool):
            """
            Processes individual data samples to transfer them to the device and convert to torch tensors if needed.

            Args:
                data: The data sample to process.
                is_validation: A flag indicating if this is a validation dataset.
                device: The device to transfer the data to.

            Returns:
                The data as torch tensors on the desired device.
            """
            return x[0] if is_validation else x  # Assume data[0] contains the inputs and data[1] the labels

        for x in dataset():
            yield process_data(x, is_validation)

