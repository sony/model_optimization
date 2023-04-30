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

import importlib

# Supported frameworks in MCT:
TENSORFLOW = 'tensorflow'
PYTORCH = 'pytorch'
FOUND_TF = importlib.util.find_spec(TENSORFLOW) is not None and importlib.util.find_spec(
    "tensorflow_model_optimization") is not None
FOUND_TORCH = importlib.util.find_spec("torch") is not None
FOUND_ONNX = importlib.util.find_spec("onnx") is not None
FOUND_ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None


# TP Model constants
OPS_SET_LIST = 'ops_set_list'

# Version
LATEST = 'latest'
