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
import random

import operator
from typing import List, Any, Tuple
import numpy as np
import torch
from torch.nn import Hardswish, Hardsigmoid, ReLU, Hardtanh, ReLU6, LeakyReLU, PReLU, SiLU, Softmax, \
    Sigmoid, Softplus, Softsign, Tanh
from torch.nn.functional import hardswish, hardsigmoid, relu, hardtanh, relu6, leaky_relu, prelu, silu, softmax, \
    softplus, softsign
from torch.nn import UpsamplingBilinear2d, AdaptiveAvgPool2d, AvgPool2d, MaxPool2d
from torch.nn.functional import upsample_bilinear, adaptive_avg_pool2d, avg_pool2d, max_pool2d
from torch.nn import Conv2d, ConvTranspose2d, Linear, BatchNorm2d
from torch.nn import Dropout, Flatten
from torch import add, multiply, mul, sub, flatten, reshape, split, unsqueeze, concat, cat,\
    mean, dropout, sigmoid, tanh
from torch.fx import symbolic_trace
from torch.nn import Module

from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.ptq import pytorch_post_training_quantization
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit.core.pytorch.constants import CALL_FUNCTION, OUTPUT, CALL_METHOD, PLACEHOLDER
from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder
from model_compression_toolkit.core.pytorch.utils import torch_tensor_to_numpy, to_torch_tensor
from tests.common_tests.base_layer_test import BaseLayerTest, LayerTestMode
from model_compression_toolkit.core.common.framework_info import set_fw_info, get_fw_info
from model_compression_toolkit.core.pytorch.default_framework_info import PyTorchInfo
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc


PYTORCH_LAYER_TEST_OPS = {
    "kernel_ops": [Conv2d, Linear, ConvTranspose2d],

    "no_quantization": [Dropout, Flatten, dropout, flatten, split, operator.getitem, reshape,
                        unsqueeze],

    "activation": [DummyPlaceHolder,
                   Hardswish, Hardsigmoid, ReLU, Hardtanh, ReLU6, LeakyReLU, PReLU, SiLU, Softmax,
                   Sigmoid, Softplus, Softsign, Tanh, hardswish, hardsigmoid, relu, hardtanh,
                   relu6, leaky_relu, prelu,
                   silu, softmax, sigmoid, softplus, softsign, tanh, torch.relu,
                   UpsamplingBilinear2d, AdaptiveAvgPool2d, AvgPool2d, MaxPool2d,
                   upsample_bilinear, adaptive_avg_pool2d, avg_pool2d, max_pool2d,
                   add, sub, mul, multiply,
                   operator.add, operator.sub, operator.mul,
                   BatchNorm2d, concat, cat, mean]
}


def seed_everything(seed_value: int):
    """
    Use seed to disable random behaviour. This is needed for the conv2dtranspose layer test, since when
    using cuda, some optimizations may cause a bit different predictions (for the same input).
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LayerTestModel(torch.nn.Module):
    def __init__(self, layer):
        super(LayerTestModel, self).__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


class OperationTestModel(torch.nn.Module):
    def __init__(self, layer):
        super(OperationTestModel, self).__init__()
        self.layer = layer

    def forward(self, x, y):
        return self.layer(x, y)


def is_node_fake_quant(node):
    return node.target == torch.fake_quantize_per_tensor_affine


def has_nested_attr(obj, attr):
    """
    Check if an object `obj` has a nested attribute `attr`.
    This function was created to test pytorch layer tests when layers are wrapped using MCTQ
    PytorchQuantizationWrapper and the internal layer should be checked and accessed.
    """
    try:
        for part in attr.split('.'):
            obj = getattr(obj, part)
        return True
    except Exception:
        return False

def get_nested_attr(obj, attr, default=None):
    """
    Retrieve the value of a nested attribute from an object `obj` if it exists,
    otherwise return a default value.
    This function was created to test pytorch layer tests when layers are wrapped using MCTQ
    PytorchQuantizationWrapper and the internal layer should be checked and accessed.
    """
    try:
        for part in attr.split('.'):
            obj = getattr(obj, part)
        return obj
    except Exception:
        return default


def get_node_operation(node, model):
    if hasattr(model, str(node.target)):
        op = getattr(model, node.target)
    elif has_nested_attr(model, node.target):
        op = get_nested_attr(model, node.target)
    elif node.op == CALL_FUNCTION:
        op = node.target
    elif node.op == CALL_METHOD:
        op = getattr(torch, node.target)
    elif node.op == PLACEHOLDER:
        op = DummyPlaceHolder
    elif node.op == OUTPUT:
        op = OUTPUT
    else:
        op = None
    return op


def get_layer_weights(layer):
    # extract layer weights and named buffers
    weights = {}
    named_parameters_weights = {name: torch_tensor_to_numpy(parameter) for name, parameter in
                                layer.named_parameters()}
    named_buffer_weights = {name: torch_tensor_to_numpy(parameter) for name, parameter in
                            layer.named_buffers() if len(parameter.shape) > 0}
    weights.update(named_parameters_weights)
    weights.update(named_buffer_weights)
    return weights


class BasePytorchLayerTest(BaseLayerTest):
    def __init__(self,
                 unit_test,
                 layers: List[Any],
                 val_batch_size: int = 1,
                 num_calibration_iter: int = 1,
                 num_of_inputs: int = 1,
                 input_shape: Tuple[int, int, int] = (3, 8, 8),
                 quantization_modes: List[LayerTestMode] = [LayerTestMode.FLOAT, LayerTestMode.QUANTIZED_8_BITS],
                 is_inputs_a_list: bool = False,
                 use_cpu: bool = False):

        super().__init__(unit_test=unit_test,
                         layers=layers,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape,
                         quantization_modes=quantization_modes,
                         is_inputs_a_list=is_inputs_a_list,
                         use_cpu=use_cpu)
        set_fw_info(PyTorchInfo)

    def get_tpc(self):
        if self.current_mode == LayerTestMode.FLOAT:
            # Disable all features that are enabled by default:
            tp = generate_test_tpc({'enable_weights_quantization': False,
                                         'enable_activation_quantization': False})
            return generate_pytorch_tpc(name="base_layer_test", tpc=tp)
        elif self.current_mode == LayerTestMode.QUANTIZED_8_BITS:
            tp = generate_test_tpc({'weights_n_bits': 8,
                                         'activation_n_bits': 8})
            return generate_pytorch_tpc(name="8bit_layer_test", tpc=tp)
        else:
            raise NotImplemented

    def get_fw_info(self) -> type[FrameworkInfo]:
        return get_fw_info()

    def get_fw_impl(self) -> FrameworkImplementation:
        return PytorchImplementation()

    def get_ptq_facade(self):
        return pytorch_post_training_quantization

    def generate_inputs(self):
        return to_torch_tensor([torch.randn(*in_shape) for in_shape in self.get_input_shapes()])

    def predict(self, model, input_tensor):
        if torch.cuda.is_available() and not self.use_cpu:
            device = 'cuda'
        else:
            device = 'cpu'
        model.to(device)
        input_tensor = [input_tensor] if not isinstance(input_tensor, list) else input_tensor
        input_tensor = [t.to(device) for t in input_tensor]
        return model(*input_tensor)

    def create_networks(self):
        models = []
        for layer in self.get_layers():
            if self.num_of_inputs > 1:
                _model = OperationTestModel(layer)
                _model.eval()
                models.append(_model)
            else:
                _model = LayerTestModel(layer)
                _model.eval()
                models.append(_model)
        return models

    def compare(self, quantized_model: Module, float_model: Module, input_x=None, quantization_info=None):
        quantized_model_fx = symbolic_trace(quantized_model)
        # Assert things that should happen when using FLOAT quantization mode
        if self.current_mode == LayerTestMode.FLOAT:
            self.__compare_float_mode(float_model, quantized_model, quantized_model_fx)

        # Assert things that should happen when using QUANTIZED_8_BITS quantization mode
        elif self.current_mode == LayerTestMode.QUANTIZED_8_BITS:
            self.__compare_8bits_quantization_mode(float_model, quantized_model, quantized_model_fx)

        # Check inference is possible
        input_tensors = self.generate_inputs()
        self.predict(quantized_model, input_tensors)
        self.predict(quantized_model_fx, input_tensors)

    def __compare_8bits_quantization_mode(self, float_model, quantized_model, quantized_model_fx):
        fw_info = self.get_fw_info()
        for node in quantized_model_fx.graph.nodes:
            op = get_node_operation(node, quantized_model)
            if op == OUTPUT or op == operator.getitem or is_node_fake_quant(node):
                continue
            if has_nested_attr(quantized_model, str(node.target)):
                if type(op) in PYTORCH_LAYER_TEST_OPS['kernel_ops']:
                    quantized_weights = get_layer_weights(get_nested_attr(quantized_model, node.target))
                    # Extract the original layer name from a wrapped PytorchQuantizationWrapper layer.
                    # For example: for conv1.layer (when conv1 is wrapping an internal layer), float_layer_name will be conv1
                    float_layer_name = str(node.target).split('.')[0]
                    float_weights = get_layer_weights(getattr(float_model, float_layer_name))
                    for k, v in quantized_weights.items():
                        # TODO: remove use of kernel op dict
                        if k == fw_info.get_kernel_op_attribute(type(op)):
                            float_weight = float_weights.get(k)
                            self.unit_test.assertFalse(float_weight is None)
                            self.unit_test.assertTrue(np.sum(np.abs(v - float_weight)) > 0.0)
                    node_next = node.next
                    while get_node_operation(node_next, quantized_model) == operator.getitem:
                        node_next = node_next.next
                    self.unit_test.assertTrue(is_node_fake_quant(node_next))

            elif op in PYTORCH_LAYER_TEST_OPS['activation']:
                node_next = node.next
                while get_node_operation(node_next, quantized_model) == operator.getitem:
                    node_next = node_next.next
                self.unit_test.assertTrue(is_node_fake_quant(node_next))

            elif op in PYTORCH_LAYER_TEST_OPS['no_quantization']:
                node_next = node.next
                while get_node_operation(node_next, quantized_model) == operator.getitem:
                    node_next = node_next.next
                self.unit_test.assertFalse(is_node_fake_quant(node_next))
            else:
                raise Exception(f'Layer {op} is not in framework info')

    def __compare_float_mode(self, float_model, quantized_model, quantized_model_fx):
        for node in quantized_model_fx.graph.nodes:
            # Check there are no fake-quant layers
            self.unit_test.assertFalse(is_node_fake_quant(node))
            # check unchanged weights
            if hasattr(quantized_model, str(node.target)):
                quantized_weights = get_layer_weights(getattr(quantized_model, node.name))
                float_weights = get_layer_weights(getattr(float_model, node.name))
                for k, v in quantized_weights.items():
                    float_weight = float_weights.get(k)
                    self.unit_test.assertFalse(float_weight is None)
                    self.unit_test.assertTrue(np.sum(np.abs(v - float_weight)) == 0.0)
        input_tensors = self.generate_inputs()

        # Set seed to avoid randomness between predictions.
        seed_everything(0)
        y = self.predict(float_model, input_tensors)
        seed_everything(0)
        y_hat = self.predict(quantized_model, input_tensors)
        if isinstance(y, (list, tuple)):
            for fo, qo in zip(y, y_hat):
                distance = torch_tensor_to_numpy(torch.sum(torch.abs(fo - qo)))
                self.unit_test.assertTrue(distance == 0,
                                          msg=f'Outputs should be identical. Observed distance: {distance}')

        else:
            distance = torch_tensor_to_numpy(torch.sum(torch.abs(y - y_hat)))
            self.unit_test.assertTrue(distance == 0,
                                      msg=f'Outputs should be identical. Observed distance: {distance}')
