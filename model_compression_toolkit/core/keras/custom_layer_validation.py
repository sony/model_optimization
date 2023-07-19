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

import inspect

def is_keras_custom_layer(layer_class: type) -> bool:
    """
    Check whether a layer class is from keras/tensorflow modules.

    Args:
        layer_class: Layer class to check its root module.

    Returns:
        Whether a layer class is from keras/tensorflow modules or not.

    """
    # Get the root module name the layer is from:
    root_module_name = inspect.getmodule(layer_class).__name__.split('.')[0]
    return root_module_name not in ['keras', 'tensorflow']
