import unittest

import torch
from torch.nn import Conv2d

import model_compression_toolkit as mct
from model_compression_toolkit.constants import THRESHOLD
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.pytorch.constants import KERNEL
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.gptq.pytorch.quantizer.soft_rounding.symmetric_soft_quantizer import \
    SymmetricSoftRoundingGPTQ

from model_compression_toolkit.quantizers_infrastructure import TrainableQuantizerWeightsConfig, \
    PytorchQuantizationWrapper

tp = mct.target_platform


class model_test(torch.nn.Module):
    def __init__(self, num_channels=3, kernel_size=1):
        super(model_test, self).__init__()
        self.conv = Conv2d(1, num_channels, kernel_size=kernel_size, bias=False)

    def forward(self, inp):
        x = self.conv(inp)
        return x


def wrap_test_model(model, sq):
    setattr(model, 'conv', PytorchQuantizationWrapper(model.conv, {KERNEL: sq}))


class TestGPTQSoftQuantizer(unittest.TestCase):

    def test_soft_targets_symmetric_per_tensor(self):
        input = to_torch_tensor(torch.ones([1, 1, 1, 1]))
        in_model = model_test().to(input.device)

        tqwc = TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod.SYMMETRIC,
                                               weights_n_bits=8,
                                               weights_quantization_params={THRESHOLD: 2.0},
                                               enable_weights_quantization=True,
                                               weights_channels_axis=1,
                                               weights_per_channel_threshold=False,
                                               min_threshold=0)

        sq = SymmetricSoftRoundingGPTQ(quantization_config=tqwc,
                                       quantization_parameter_learning=False)
        wrap_test_model(in_model, sq)

        conv_wrap_layer = [m for m in in_model.modules() if isinstance(m, PytorchQuantizationWrapper)][0]
        float_weights = [x[1] for x in conv_wrap_layer._weights_vars if x[0] == KERNEL][0]

        in_model.eval()
        out = in_model(input)
        self.assertTrue(torch.any(float_weights != out.reshape(float_weights.shape)))

        in_model.train()
        out_t = in_model(input)
        self.assertTrue(torch.all(float_weights == out_t.reshape(float_weights.shape)))

    def test_soft_targets_symmetric_per_channel(self):
        input = to_torch_tensor(torch.ones([1, 1, 2, 2]))
        in_model = model_test(num_channels=1, kernel_size=2).to(input.device)

        tqwc = TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod.SYMMETRIC,
                                               weights_n_bits=8,
                                               weights_quantization_params={THRESHOLD: 2.0},
                                               enable_weights_quantization=True,
                                               weights_channels_axis=1,
                                               weights_per_channel_threshold=True,
                                               min_threshold=0)

        sq = SymmetricSoftRoundingGPTQ(quantization_config=tqwc,
                                       quantization_parameter_learning=False)
        wrap_test_model(in_model, sq)

        conv_wrap_layer = [m for m in in_model.modules() if isinstance(m, PytorchQuantizationWrapper)][0]
        float_weights = [x[1] for x in conv_wrap_layer._weights_vars if x[0] == KERNEL][0]

        in_model.eval()
        out = in_model(input)
        self.assertFalse(torch.isclose(torch.sum(float_weights), out))

        in_model.train()
        out_t = in_model(input)
        self.assertTrue(torch.isclose(torch.sum(float_weights), out_t))