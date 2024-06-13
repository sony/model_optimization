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

from typing import Callable

from model_compression_toolkit.logger import Logger


class DatasetUtils:
    """
    Class with helpful methods for handling different kinds of datasets from the user.
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

        Logger.critical("This method should be implemented by the framework-specific DatasetUtils.")  # pragma: no cover

