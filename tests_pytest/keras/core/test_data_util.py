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
from model_compression_toolkit.core.keras.data_util import create_dataloader_from_data_generator
from tests_pytest.common.test_data_util_base import TestDataLoaderFromGeneratorBase


class TestDataloaderFromGenerator(TestDataLoaderFromGeneratorBase):
    create_dataloader_fn = create_dataloader_from_data_generator
    __test__ = True
