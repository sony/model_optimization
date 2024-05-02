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
from typing import Tuple

import torch

from model_compression_toolkit import logger
from model_compression_toolkit.core.pytorch.constants import CUDA, CPU


class DeviceManager:
    """
    A singleton class to manage the PyTorch device (CPU or CUDA) across different parts of the project.
    Ensures that only one instance of this class is created and used throughout all files.
    """

    _instance = None

    def __new__(cls):
        """
        Override the __new__ method to implement the singleton pattern.
        Ensures that only one instance of DeviceManager is created.
        """
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            # Initialize the default device
            cls._instance.DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)
        return cls._instance

    def set_device(self, device_name: str):
        """
        Set the device for PyTorch operations.

        Args:
            device_name (str): The name of the device, e.g., 'cuda:0' or 'cpu'.

        If the specified device is not valid or available, it prints a warning message without changing the current device.
        """
        is_valid, message = self.is_valid_device(device_name)
        if is_valid:
            self.DEVICE = torch.device(device_name)
        else:
            logger.Logger.warning(message)

    def get_device(self) -> torch.device:
        """
        Get the current PyTorch device.

        Returns:
            torch.device: The current device set for PyTorch operations.
        """
        return self.DEVICE

    @staticmethod
    def is_valid_device(device_name: str) -> Tuple[bool, str]:
        """
        Check if the specified device name is valid and available.

        Args:
            device_name (str): The name of the device to check, e.g., 'cuda:0' or 'cuda'.

        Returns:
            tuple: (bool, str) A tuple where the first element is a boolean indicating if the device is valid,
                   and the second element is a message describing the validity or the issue.
        """
        if CUDA in device_name:
            if not torch.cuda.is_available():
                return False, "CUDA is not available"

            if device_name == CUDA:
                # 'cuda' without a specific index is valid if CUDA is available
                return True, "Valid device"

            try:
                device_index = int(device_name.split(':')[1])
                if device_index >= torch.cuda.device_count():
                    return False, f"CUDA device index {device_index} out of range. Number of valid devices: {torch.cuda.device_count()}"
            except Exception:
                # Handle cases where the device name is incorrectly formatted
                return False, "Invalid CUDA device format. Use 'cuda' or 'cuda:x' where x is the device index."

            return True, "Valid device"

        if CPU in device_name:
            return True, "Valid device"

        return False, "Invalid device"



def set_working_device(device_name: str):
    """
    Set the device for PyTorch operations.

    Args:
        device_name (str): The name of the device, e.g., 'cuda:0' or 'cpu'.

    If the specified device is not valid or available, it prints an error message without changing the current device.
    """
    device_manager = DeviceManager()
    device_manager.set_device(device_name)

def get_working_device() -> torch.device:
    """
    Get the current PyTorch device.

    Returns:
        torch.device: The current device set for PyTorch operations.
    """
    device_manager = DeviceManager()
    return device_manager.get_device()