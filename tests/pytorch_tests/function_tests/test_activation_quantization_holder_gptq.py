import copy
import unittest

import numpy as np
import torch
from torch.nn import Conv2d

import model_compression_toolkit as mct
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchQuantizationWrapper
from model_compression_toolkit.core.common.mixed_precision.bit_width_setter import set_bit_widths
from model_compression_toolkit.gptq import GradualActivationQuantizationConfig, QFractionLinearAnnealingConfig
from model_compression_toolkit.gptq.pytorch.gptq_pytorch_implementation import GPTQPytorchImplemantation
from model_compression_toolkit.gptq.pytorch.gptq_training import PytorchGPTQTrainer
from model_compression_toolkit.gptq.common.gradual_activation_quantization import \
    GradualActivationQuantizerWrapper
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit.trainable_infrastructure import TrainingMethod
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.trainable_infrastructure.pytorch.activation_quantizers import \
    STESymmetricActivationTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure.pytorch.annealing_schedulers import PytorchLinearAnnealingScheduler
from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.core.pytorch.default_framework_info import PyTorchInfo
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_quantization_parameters
from tests.pytorch_tests.utils import get_layers_from_model_by_type

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
    def setUp(self):
        set_fw_info(PyTorchInfo)

    def test_adding_holder_instead_quantize_wrapper(self):
        gptq_model = self._get_gptq_model(INPUT_SHAPE, BasicModel())
        activation_quantization_holders_in_model = self._get_holders_with_validation(gptq_model, exp_n_holders=3)
        for a in activation_quantization_holders_in_model:
            self.assertTrue(isinstance(a.activation_holder_quantizer, STESymmetricActivationTrainableQuantizer))
            self.assertEqual(a.activation_holder_quantizer.identifier, TrainingMethod.STE)
            # activation quantization params for gptq should be frozen (non-learnable)
            self.assertTrue(a.activation_holder_quantizer.freeze_quant_params is True)
            self.assertEqual(a.activation_holder_quantizer.get_trainable_variables(VariableGroup.QPARAMS), [])

        for name, module in gptq_model.named_modules():
            if isinstance(module, PytorchQuantizationWrapper):
                self.assertTrue(len(module.weights_quantizers) > 0)

    def test_adding_holder_after_relu(self):
        gptq_model = self._get_gptq_model(INPUT_SHAPE, ReLUModel())
        activation_quantization_holders_in_model = self._get_holders_with_validation(gptq_model, exp_n_holders=3)
        for a in activation_quantization_holders_in_model:
            self.assertTrue(isinstance(a.activation_holder_quantizer, STESymmetricActivationTrainableQuantizer))
        for name, module in gptq_model.named_modules():
            if isinstance(module, PytorchQuantizationWrapper):
                self.assertTrue(len(module.weights_quantizers) > 0)

    def test_adding_holders_after_reuse(self):
        float_model = ReuseModel()
        gptq_model = self._get_gptq_model(INPUT_SHAPE, float_model)
        activation_quantization_holders_in_model = self._get_holders_with_validation(gptq_model, exp_n_holders=3)
        for a in activation_quantization_holders_in_model:
            self.assertTrue(isinstance(a.activation_holder_quantizer, STESymmetricActivationTrainableQuantizer))
        for name, module in gptq_model.named_modules():
            if isinstance(module, PytorchQuantizationWrapper):
                self.assertTrue(len(module.weights_quantizers) > 0)

        self.assertEqual([p.data_ptr() for p in gptq_model.conv.parameters()],
                         [p.data_ptr() for p in gptq_model.conv_1.parameters()],
                         f"Shared parameters between reused layers should have identical memory addresses")

    def test_adding_holder_with_gradual_act_quantization(self):
        gradual_act_quant_cfg = GradualActivationQuantizationConfig(
            QFractionLinearAnnealingConfig(initial_q_fraction=0.1, target_q_fraction=0.9, start_step=100, end_step=500)
        )
        gptq_cfg = mct.gptq.get_pytorch_gptq_config(1, use_hessian_based_weights=False,
                                                    use_hessian_sample_attention=False,
                                                    gradual_activation_quantization=gradual_act_quant_cfg)
        gptq_model = self._get_gptq_model(INPUT_SHAPE, BasicModel(), gptq_cfg)
        activation_holders = self._get_holders_with_validation(gptq_model, exp_n_holders=3)

        for a in activation_holders:
            self.assertTrue(isinstance(a.activation_holder_quantizer, GradualActivationQuantizerWrapper))
            # check that quantizer wrapper's scheduler was created according to gptq config
            factor_scheduler = a.activation_holder_quantizer.q_fraction_scheduler
            self.assertTrue(isinstance(factor_scheduler, PytorchLinearAnnealingScheduler))
            self.assertEqual(factor_scheduler.t_start, 100)
            self.assertEqual(factor_scheduler.t_end, 500)
            self.assertEqual(factor_scheduler.initial_val, 0.1)
            self.assertEqual(factor_scheduler.target_val, 0.9)
            # check the wrapped quantizer is correct and frozen
            quantizer = a.activation_holder_quantizer.quantizer
            self.assertTrue(isinstance(quantizer, STESymmetricActivationTrainableQuantizer))
            self.assertTrue(quantizer.freeze_quant_params is True)
            self.assertEqual(quantizer.get_trainable_variables(VariableGroup.QPARAMS), [])

    def _get_holders_with_validation(self, gptq_model, exp_n_holders):
        last_module = list(gptq_model.named_modules())[-1][1]
        activation_quantization_holders = get_layers_from_model_by_type(gptq_model, PytorchActivationQuantizationHolder)
        # the last module should be an activation quantization holder
        self.assertTrue(isinstance(last_module, PytorchActivationQuantizationHolder))
        self.assertTrue(len(activation_quantization_holders) == exp_n_holders)
        return activation_quantization_holders

    def _get_gptq_model(self, input_shape, in_model, gptq_cfg=None):
        pytorch_impl = GPTQPytorchImplemantation()
        qc = copy.deepcopy(mct.core.DEFAULTCONFIG)
        qc.linear_collapsing = False
        graph = prepare_graph_with_quantization_parameters(in_model,
                                                           pytorch_impl,
                                                           representative_dataset,
                                                           generate_pytorch_tpc,
                                                           [1] + input_shape,
                                                           mixed_precision_enabled=False,
                                                           qc=qc,
                                                           attach2fw=AttachTpcToPytorch())
        graph = set_bit_widths(mixed_precision_enable=False,
                               graph=graph)
        gptq_cfg = gptq_cfg or mct.gptq.get_pytorch_gptq_config(1, use_hessian_based_weights=False,
                                                                use_hessian_sample_attention=False,
                                                                gradual_activation_quantization=False)
        trainer = PytorchGPTQTrainer(graph,
                                     graph,
                                     gptq_cfg,
                                     pytorch_impl,
                                     representative_dataset)
        gptq_model, _ = trainer.build_gptq_model()
        return gptq_model
