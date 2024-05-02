#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import unittest
import numpy as np
import torch
from unittest.mock import patch

from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy


class TestTensorConversions(unittest.TestCase):
    def setUp(self):
        # Setup any common resources
        self.numpy_array = np.array([1.0, 2.0, 3.0])
        self.torch_tensor = torch.tensor([1.0, 2.0, 3.0])
        self.list_of_numbers = [1, 2, 3]
        self.tuple_of_numbers = (1, 2, 3)

    @patch('model_compression_toolkit.core.pytorch.pytorch_device_config.get_working_device')
    def test_to_torch_tensor_with_numpy_array(self, mock_get_device):
        mock_get_device.return_value = 'cpu'
        result = to_torch_tensor(self.numpy_array)
        self.assertTrue(torch.equal(result.to('cpu'), torch.tensor([1.0, 2.0, 3.0], device='cpu')))
        self.assertIsInstance(result, torch.Tensor)

    @patch('model_compression_toolkit.core.pytorch.pytorch_device_config.get_working_device')
    def test_to_torch_tensor_with_torch_tensor(self, mock_get_device):
        mock_get_device.return_value = 'cpu'
        result = to_torch_tensor(self.torch_tensor)
        self.assertTrue(torch.equal(result.to('cpu'), self.torch_tensor.to('cpu')))

    @patch('model_compression_toolkit.core.pytorch.pytorch_device_config.get_working_device')
    def test_to_torch_tensor_with_list(self, mock_get_device):
        mock_get_device.return_value = 'cpu'
        result = to_torch_tensor(self.list_of_numbers)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(x, torch.Tensor) for x in result))

    @patch('model_compression_toolkit.core.pytorch.pytorch_device_config.get_working_device')
    def test_to_torch_tensor_with_tuple(self, mock_get_device):
        mock_get_device.return_value = 'cpu'
        result = to_torch_tensor(self.tuple_of_numbers)
        self.assertEqual(len(tuple(result)), 3)
        self.assertTrue(all(isinstance(x, torch.Tensor) for x in result))

    @patch('model_compression_toolkit.core.pytorch.pytorch_device_config.get_working_device')
    @patch('model_compression_toolkit.logger.Logger')
    def test_to_torch_tensor_with_unsupported_type(self, mock_logger, mock_get_device):
        with self.assertRaises(Exception):
            mock_get_device.return_value = 'cpu'
            to_torch_tensor("unsupported_type")
            mock_logger.critical.assert_called_once()

    def test_torch_tensor_to_numpy_with_torch_tensor(self):
        result = torch_tensor_to_numpy(self.torch_tensor)
        np.testing.assert_array_almost_equal(result, self.numpy_array)

    def test_torch_tensor_to_numpy_with_list(self):
        result = torch_tensor_to_numpy([self.torch_tensor, self.torch_tensor])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(x, np.ndarray) for x in result))

    def test_torch_tensor_to_numpy_with_tuple(self):
        result = torch_tensor_to_numpy((self.torch_tensor, self.torch_tensor))
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(x, np.ndarray) for x in result))

    @patch('model_compression_toolkit.logger.Logger')
    def test_torch_tensor_to_numpy_with_unsupported_type(self, mock_logger):
        with self.assertRaises(Exception):
            torch_tensor_to_numpy("unsupported_type")
            mock_logger.critical.assert_called_once()

if __name__ == '__main__':
    unittest.main()

