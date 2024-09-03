import copy
import unittest

import numpy as np
import torch
from torch.nn import Conv2d

import model_compression_toolkit as mct
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchQuantizationWrapper
from model_compression_toolkit.core.common.mixed_precision.bit_width_setter import set_bit_widths
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.gptq.pytorch.gptq_pytorch_implementation import GPTQPytorchImplemantation
from model_compression_toolkit.gptq.pytorch.gptq_training import PytorchGPTQTrainer
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit.trainable_infrastructure import TrainingMethod
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.trainable_infrastructure.pytorch.activation_quantizers import \
    STESymmetricActivationTrainableQuantizer
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_quantization_parameters

INPUT_SHAPE = [3, 8, 8]


class BasicModel(torch.nn.Module):
    def __init__(self, num_channels=3, kernel_size=1):
        super(BasicModel, self).__init__()
        self.conv1 = Conv2d(num_channels, num_channels, kernel_size=kernel_size, bias=False)
        self.conv2 = Conv2d(num_channels, num_channels, kernel_size=kernel_size, bias=False)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.conv2(x)
        return x


class ReLUModel(torch.nn.Module):
    def __init__(self, num_channels=3, kernel_size=1):
        super(ReLUModel, self).__init__()
        self.conv1 = Conv2d(num_channels, num_channels, kernel_size=kernel_size, bias=False)
        self.conv2 = Conv2d(num_channels, num_channels, kernel_size=kernel_size, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class ReuseModel(torch.nn.Module):
    def __init__(self, num_channels=3, kernel_size=1):
        super(ReuseModel, self).__init__()
        self.conv = Conv2d(num_channels, num_channels, kernel_size=kernel_size, bias=False)

    def forward(self, inp):
        x = self.conv(inp)
        x = self.conv(x)
        return x


def representative_dataset():
    yield [np.random.randn(*[1] + INPUT_SHAPE).astype(np.float32)]


class TestGPTQModelBuilderWithActivationHolder(unittest.TestCase):

    def test_adding_holder_instead_quantize_wrapper(self):
        gptq_model = self._get_gptq_model(INPUT_SHAPE, BasicModel())
        last_module = list(gptq_model.named_modules())[-1][1]
        activation_quantization_holders_in_model = [m[1] for m in gptq_model.named_modules() if isinstance(m[1], PytorchActivationQuantizationHolder)]
        # the last module should be an activation quantization holder
        self.assertTrue(isinstance(last_module, PytorchActivationQuantizationHolder))
        # check that 4 activation quantization holders where generated
        self.assertTrue(len(activation_quantization_holders_in_model) == 3)
        for a in activation_quantization_holders_in_model:
            self.assertTrue(isinstance(a.activation_holder_quantizer, STESymmetricActivationTrainableQuantizer))
            self.assertEquals(a.activation_holder_quantizer.identifier, TrainingMethod.STE)
            # activation quantization params for gptq should be frozen (non-learnable)
            self.assertTrue(a.activation_holder_quantizer.freeze_quant_params is True)
            self.assertEquals(a.activation_holder_quantizer.get_trainable_variables(VariableGroup.QPARAMS), [])

        for name, module in gptq_model.named_modules():
            if isinstance(module, PytorchQuantizationWrapper):
                self.assertTrue(len(module.weights_quantizers) > 0)

    def test_adding_holder_after_relu(self):
        gptq_model = self._get_gptq_model(INPUT_SHAPE, ReLUModel())
        last_module = list(gptq_model.named_modules())[-1][1]
        activation_quantization_holders_in_model = [m[1] for m in gptq_model.named_modules() if isinstance(m[1], PytorchActivationQuantizationHolder)]
        # the last module should be an activation quantization holder
        self.assertTrue(isinstance(last_module, PytorchActivationQuantizationHolder))
        # check that 3 activation quantization holders where generated
        self.assertTrue(len(activation_quantization_holders_in_model) == 3)
        for a in activation_quantization_holders_in_model:
            self.assertTrue(isinstance(a.activation_holder_quantizer, STESymmetricActivationTrainableQuantizer))
        for name, module in gptq_model.named_modules():
            if isinstance(module, PytorchQuantizationWrapper):
                self.assertTrue(len(module.weights_quantizers) > 0)

    def test_adding_holders_after_reuse(self):
        float_model = ReuseModel()
        gptq_model = self._get_gptq_model(INPUT_SHAPE, float_model)
        activation_quantization_holders_in_model = [m[1] for m in gptq_model.named_modules() if isinstance(m[1], PytorchActivationQuantizationHolder)]
        last_module = list(gptq_model.named_modules())[-1][1]
        # the last module should be an activation quantization holder
        self.assertTrue(isinstance(last_module, PytorchActivationQuantizationHolder))
        # check that 4 activation quantization holders where generated
        self.assertTrue(len(activation_quantization_holders_in_model) == 3)
        for a in activation_quantization_holders_in_model:
            self.assertTrue(isinstance(a.activation_holder_quantizer, STESymmetricActivationTrainableQuantizer))
        for name, module in gptq_model.named_modules():
            if isinstance(module, PytorchQuantizationWrapper):
                self.assertTrue(len(module.weights_quantizers) > 0)
        # Test that two holders are getting inputs from reused conv2d (the layer that is wrapped)

        # FIXME there is no reuse support and the test doesn't test what it says it tests. It doesn't even look
        # at correct layers. After moving to trainable quantizer the test makes even less sense since now fx traces
        # all quantization operations instead of fake_quant layer.
        # fx_model = symbolic_trace(gptq_model)
        # self.assertTrue(list(fx_model.graph.nodes)[3].all_input_nodes[0] == list(fx_model.graph.nodes)[2])
        # self.assertTrue(list(fx_model.graph.nodes)[6].all_input_nodes[0] == list(fx_model.graph.nodes)[5])

    def _get_gptq_model(self, input_shape, in_model):
        pytorch_impl = GPTQPytorchImplemantation()
        qc = copy.deepcopy(mct.core.DEFAULTCONFIG)
        qc.linear_collapsing = False
        graph = prepare_graph_with_quantization_parameters(in_model,
                                                           pytorch_impl,
                                                           DEFAULT_PYTORCH_INFO,
                                                           representative_dataset,
                                                           generate_pytorch_tpc,
                                                           [1] + input_shape,
                                                           mixed_precision_enabled=False,
                                                           qc=qc)
        graph = set_bit_widths(mixed_precision_enable=False,
                               graph=graph)
        trainer = PytorchGPTQTrainer(graph,
                                   graph,
                                   mct.gptq.get_pytorch_gptq_config(1, use_hessian_based_weights=False),
                                   pytorch_impl,
                                   DEFAULT_PYTORCH_INFO,
                                   representative_dataset)
        gptq_model, _ = trainer.build_gptq_model()
        return gptq_model