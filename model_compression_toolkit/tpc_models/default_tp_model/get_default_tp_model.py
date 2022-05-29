# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from typing import Any
from model_compression_toolkit.common.target_platform import TargetPlatformModel
from model_compression_toolkit.common.logger import Logger
import pkgutil
import importlib
import os

def import_model(version: str = None) -> Any:
    """
    Import specific Target Platform module by version
    Args
        version: Specific version (i.e '3.2.0'), None for latest version
    Returns: An import model
    """
    if version is None:
        # Take latest
        version_name = max([name for _, name, _ in pkgutil.iter_modules([os.path.dirname(__file__)])])
    else:
        # Specific version: expected format: X.X.X
        version_name = 'v'+version.replace('.','_')

    module_name = '.'.join(__name__.split('.')[:-1])+'.'+version_name
    import_model = None
    try:
        import_model = importlib.import_module(module_name)
    except ImportError:
        Logger.error("No version {} found!".format(version_name))

    return import_model


def get_default_tp_model(version: str = None) -> TargetPlatformModel:
    """
    A method that generates a default target platform model, with base 8-bit quantization configuration and 8, 4, 2
    bits configuration list for mixed-precision quantization.
    NOTE: in order to generate a target platform model with different configurations but with the same Operators Sets
    (for tests, experiments, etc.), use this method implementation as a test-case, i.e., override the
    'get_op_quantization_configs' method and use its output to call 'generate_tp_model' with your configurations.
    Args
        version: Specific version (i.e '3.2.0'), None for latest version
    Returns: A TargetPlatformModel object.
    """
    m = import_model(version)
    base_config, mixed_precision_cfg_list = m.get_op_quantization_configs()
    return m.generate_tp_model(default_config=base_config,
                             base_config=base_config,
                             mixed_precision_cfg_list=mixed_precision_cfg_list,
                             name='default_tp_model')
