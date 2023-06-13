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
from abc import abstractmethod, ABC


class BaseModelLib(ABC):
    """
    Abstract base class representing pre-trained model library.

    Args:
        args: Arguments required for initializing the model library and selecting the relevant model

    Attributes:
        args: The input arguments
    """

    def __init__(self, args):
        self.args = args

    @abstractmethod
    def get_model(self):
        """
        Abstract method to return the floating-point model for quantization.

        Returns:
            The floating-point model to be used for quantization.
        """
        raise NotImplementedError

    @abstractmethod
    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size):
        """
        Abstract method to return the representative dataset required by MCT for quantization.

        Args:
            representative_dataset_folder: Path to the representative dataset folder.
            n_iter: Number of calibration iterations to perform for quantization. It uses for creating the representative dataset.
            batch_size: Batch size for processing the representative dataset.

        Returns:
            The representative dataset generator to be used for quantization.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model):
        """
        Abstract method to perform evaluation for a given model.

        Args:
            model: The model to be evaluated.

        Returns:
            The evaluation results of the model.
        """
        raise NotImplementedError


class ModuleReplacer(ABC):
    """
    Abstract base class for replacing modules in a given model.

    """

    def __init__(self):
        """
        Initializes the ModuleReplacer object.
        """
        pass

    @abstractmethod
    def get_new_module(self, config):
        """
       Abstract method to implement and return the new module based on a given configuration.

       Args:
           config: Configuration for the new module.

       Returns:
           The new module based on the given configuration.
        """
        raise NotImplementedError

    @abstractmethod
    def get_config(self, old_module):
        """
        Abstract method to return the configuration of the old module.

        Args:
            old_module: The old module to retrieve the configuration from.

        Returns:
            The configuration of the old module.
        """
        raise NotImplementedError

    @abstractmethod
    def replace(self, model):
        """
        Abstract method to replace the old module with the new module in a given model.

        Args:
            model: The model in which the module replacement should be performed.

        Returns:
            The model with the replaced module.
        """
        raise NotImplementedError


