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
import unittest
from model_compression_toolkit.tpc_models.default_tp_model.get_default_tp_model import import_model


class TestTPCVersion(unittest.TestCase):

    def test_valid_import_model(self):
        versions = ['2.7.0', '2_7_0', '3.2.0', '3_2_0', None]
        for version in versions:
            _ = import_model(version)

    def test_invalid_import_model(self):
        versions = ['2.7.1', '3.2', '2_5_0', '23rs']
        for version in versions:
            ok = False
            try:
                _ = import_model(version)
            except:
                ok = True
            self.assertTrue(ok, msg="Error: Version is invalid but still we can import it!")

if __name__ == '__main__':
    unittest.main()
