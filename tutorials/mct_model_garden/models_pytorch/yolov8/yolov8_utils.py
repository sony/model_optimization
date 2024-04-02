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
from typing import List, Any

import numpy as np
import torch

from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device


def model_predict(model: Any,
                  inputs: np.ndarray) -> List:
    """
    Perform inference using the provided PyTorch model on the given inputs.

    This function handles moving the inputs to the appropriate torch device and data type,
    and detaches and moves the outputs to the CPU.

    Args:
        model (Any): The PyTorch model used for inference.
        inputs (np.ndarray): Input data to perform inference on.

    Returns:
        List: List containing tensors of predictions.
    """
    device = get_working_device()
    inputs = torch.from_numpy(inputs).to(device=device, dtype=torch.float)

    # Run Pytorch inference on the batch
    outputs = model(inputs)

    # Detach outputs and move to cpu
    output_list = [output.detach().cpu() for output in outputs]

    return output_list
