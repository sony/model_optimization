# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest


class BaseModelTest(BasePytorchTest):
    def __init__(self,
                 unit_test,
                 model,
                 float_reconstruction_error=1e-5,
                 convert_to_fx=True):
        super().__init__(unit_test, float_reconstruction_error, convert_to_fx)
        self.model = model

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 224, 224]]

    def create_feature_network(self, input_shape):
        return self.model(pretrained=True)