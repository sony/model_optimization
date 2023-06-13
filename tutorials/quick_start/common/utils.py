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
import importlib


def find_modules(lib: str) -> Tuple[str, str]:
    """
    Finds the relevant module paths for a given pre-trained models library.

    Args:
        lib (str): Name of the library.

    Returns:
        Tuple of model_lib_module and quant_module, representing the module paths needed to be imported
        for the given library.

    Raises:
        Error: If the specified library is not found.
    """

    # Search in PyTorch libraries
    if importlib.util.find_spec('pytorch_fw.' + lib) is not None:
        model_lib_module = 'pytorch_fw.' + lib + '.model_lib_' + lib
        quant_module = 'pytorch_fw.quant'

    # Search in Keras libraries
    elif importlib.util.find_spec('keras_fw.' + lib) is not None:
        model_lib_module = 'keras_fw.' + lib + '.model_lib_' + lib
        quant_module = 'keras_fw.quant'
    else:
        raise Exception(f'Error: model library {lib} is not supported')
    return model_lib_module, quant_module
