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
import torch
import torch.nn as nn
import numpy as np
import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy, set_model
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.tensors_compare import cosine_similarity

tp = mct.target_platform


class ConstRepresentationNet(nn.Module):
    def __init__(self, layer, const):
        super().__init__()
        self.layer = layer
        self.const = to_torch_tensor(const) if isinstance(const, np.ndarray) else const

    def forward(self, x):
        return self.layer(x, self.const)


class ConstRepresentationReverseOrderNet(nn.Module):
    def __init__(self, layer, const):
        super().__init__()
        self.layer = layer
        self.const = to_torch_tensor(const) if isinstance(const, np.ndarray) else const

    def forward(self, x):
        return self.layer(self.const, x)


class ConstRepresentationTest(BasePytorchFeatureNetworkTest):

    def __init__(self, unit_test, func, const, input_reverse_order=False):
        super().__init__(unit_test=unit_test, input_shape=(16, 32, 32))
        self.func = func
        self.const = const
        self.input_reverse_order = input_reverse_order

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 32,
                                     'activation_n_bits': 32,
                                     'enable_activation_quantization': False})
        return generate_pytorch_tpc(name="linear_collapsing_test", tp_model=tp)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.NOCLIPPING,
                                           mct.core.QuantizationErrorMethod.NOCLIPPING,
                                           False, False, True)

    def create_networks(self):
        if self.input_reverse_order:
            return ConstRepresentationReverseOrderNet(self.func, self.const)
        else:
            return ConstRepresentationNet(self.func, self.const)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        in_torch_tensor = to_torch_tensor(input_x[0])
        set_model(float_model)
        y = float_model(in_torch_tensor)
        y_hat = quantized_model(in_torch_tensor)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        cs = cosine_similarity(torch_tensor_to_numpy(y), torch_tensor_to_numpy(y_hat))
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')


class ConstRepresentationMultiInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.const1 = to_torch_tensor(np.random.random((32,)))
        self.const2 = to_torch_tensor(np.random.random((32,)))
        self.const3 = to_torch_tensor(np.random.random((1, 5, 32, 32)))

    def forward(self, x):
        x1 = sum(
            [self.const1, x, self.const2])  # not really a 3-input add operation, but just in case torch will support it
        x = torch.cat([x1, self.const3, x], dim=1)
        return x


class ConstRepresentationMultiInputTest(ConstRepresentationTest):

    def __init__(self, unit_test):
        super().__init__(unit_test=unit_test, func=None, const=None, input_reverse_order=False)

    def create_networks(self):
        return ConstRepresentationMultiInputNet()


class ConstRepresentationLinearLayerNet(nn.Module):
    def __init__(self, layer, const):
        super().__init__()
        self.layer = layer
        self.const = to_torch_tensor(const) if isinstance(const, np.ndarray) else const

    def forward(self, x):
        x1 = self.layer(self.const)
        x2 = x + self.const
        return x1 + x2


class ConstRepresentationLinearLayerTest(ConstRepresentationTest):

    def __init__(self, unit_test, func, const, enable_weights_quantization):
        super().__init__(unit_test=unit_test, func=func, const=const, input_reverse_order=False)
        self.enable_weights_quantization = enable_weights_quantization

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 32,
                                     'activation_n_bits': 32,
                                     'enable_weights_quantization': self.enable_weights_quantization,
                                     'enable_activation_quantization': False})
        return generate_pytorch_tpc(name="linear_collapsing_test", tp_model=tp)

    def create_networks(self):
        return ConstRepresentationLinearLayerNet(self.func, self.const)


class ConstRepresentationGetIndexNet(nn.Module):
    def __init__(self, layer, const, indices):
        super().__init__()
        self.layer = layer
        self.const = to_torch_tensor(const) if isinstance(const, np.ndarray) else const
        self.indices = indices

    def forward(self, x):
        const = self.const[self.indices]
        return self.layer(x, const)


class ConstRepresentationGetIndexTest(ConstRepresentationTest):

    def __init__(self, unit_test, func, const, indices):
        super().__init__(unit_test=unit_test, func=func, const=const, input_reverse_order=False)
        self.func = func
        self.const = const
        self.indices = indices

    def create_networks(self):
        return ConstRepresentationGetIndexNet(self.func, self.const, self.indices)


class ConstRepresentationCodeNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.conv2d = nn.Conv2d(3, 16, 3, 2, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.register_buffer('sub_const', 10 * torch.rand((1, 16, 64)))

    def forward(self, x):
        _shape = x.shape[2:]
        x = self.conv2d(x)

        # input tensor in kwargs
        x = nn.functional.interpolate(x, size=_shape)

        # reshaping batch_norm input to 3 axes to avoid bn-folding.
        x = x.reshape((-1, 16, int(np.prod(self.input_shape))))

        # input const in kwargs (not the first kwargs!)
        x = nn.functional.batch_norm(x,
                                     self.bn.running_mean, self.bn.running_var,
                                     momentum=0.2, eps=1e-6, bias=self.bn.bias)

        # input all tensors and consts in kwargs
        x = torch.sub(input=self.sub_const, other=x)

        return torch.reshape(x, (-1, 16) + self.input_shape)


class ConstRepresentationCodeTest(BasePytorchFeatureNetworkTest):

    def __init__(self, unit_test):
        super().__init__(unit_test=unit_test)

    def create_networks(self):
        return ConstRepresentationCodeNet(self.input_shape[2:])

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 32,
                                     'activation_n_bits': 32,
                                     'enable_activation_quantization': False})
        return generate_pytorch_tpc(name="linear_collapsing_test", tp_model=tp)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        in_torch_tensor = to_torch_tensor(input_x[0])
        set_model(float_model)
        y = float_model(in_torch_tensor)
        y_hat = quantized_model(in_torch_tensor)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        cs = cosine_similarity(torch_tensor_to_numpy(y), torch_tensor_to_numpy(y_hat))
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')
