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
import copy
import os
import torch
import numpy as np
import random
import unittest

from mct_quantizers import PytorchQuantizationWrapper
from mct_quantizers.pytorch.quantizers import WeightsPOTInferableQuantizer, ActivationPOTInferableQuantizer
from torchvision.models import mobilenet_v2

import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor

tp = mct.target_platform


class TestFullyQuantizedExporter(unittest.TestCase):

    def setUp(self) -> None:
        self.mbv2 = mobilenet_v2(pretrained=True)
        self.representative_data_gen = self.random_data_gen
        self.fully_quantized_mbv2 = self.run_mct(self.mbv2)

    def random_data_gen(self, n_iters=1):
        for _ in range(n_iters):
            yield to_torch_tensor([torch.randn(1, 3, 224, 224)])

    def run_mct(self, model):
        core_config = mct.core.CoreConfig()
        new_export_model, _ = mct.ptq.pytorch_post_training_quantization_experimental(
            in_module=model,
            core_config=core_config,
            representative_data_gen=self.representative_data_gen,
            new_experimental_exporter=True)
        return new_export_model

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def test_sanity(self):
        """
        Test that new fully quantized exporter model predicts the same as
        old exported model.
        """
        model = mobilenet_v2(pretrained=True)
        def repr_dataset(n_iters=1):
            for _ in range(n_iters):
                yield to_torch_tensor([torch.ones(1, 3, 224, 224)])

        seed = np.random.randint(0, 100, size=1)[0]

        self.set_seed(seed)
        core_config = mct.core.CoreConfig()
        old_export_model, _ = mct.ptq.pytorch_post_training_quantization_experimental(
            in_module=model,
            representative_data_gen=repr_dataset,
            core_config=core_config,
            new_experimental_exporter=False)

        self.set_seed(seed)
        core_config = mct.core.CoreConfig()
        new_export_model, _ = mct.ptq.pytorch_post_training_quantization_experimental(
            in_module=model,
            core_config=core_config,
            representative_data_gen=repr_dataset,
            new_experimental_exporter=True)

        def _to_numpy(t):
            return t.cpu().detach().numpy()

        images = next(repr_dataset())
        diff = new_export_model(images) - old_export_model(images)
        w_delta = np.sum(np.abs(_to_numpy(old_export_model.features_0_0_bn.weight) - _to_numpy(new_export_model.features_0_0_bn.layer.weight)))
        self.assertTrue(w_delta == 0, f'Diff between weights: {w_delta}')
        self.assertTrue(np.sum(np.abs(diff.cpu().detach().numpy())) == 0, f'Sum absolute error: {np.sum(np.abs(diff.cpu().detach().numpy()))}')


    def _to_numpy(self, t):
        return t.cpu().detach().numpy()

    def test_layers_wrapper(self):
        self.assertTrue(isinstance(self.fully_quantized_mbv2.features_0_0_bn, PytorchQuantizationWrapper))
        self.assertTrue(isinstance(self.fully_quantized_mbv2.features_0_0_bn.layer, torch.nn.Conv2d))
        self.assertTrue(self.fully_quantized_mbv2.features_0_0_bn.is_weights_quantization)

        self.assertTrue(isinstance(self.fully_quantized_mbv2.classifier_1, PytorchQuantizationWrapper))
        self.assertTrue(isinstance(self.fully_quantized_mbv2.classifier_1.layer, torch.nn.Linear))
        self.assertTrue(self.fully_quantized_mbv2.classifier_1.is_weights_quantization)

    def test_weights_qc(self):
        self.assertTrue(len(self.fully_quantized_mbv2.features_0_0_bn.weights_quantizers) == 1)
        self.assertTrue(isinstance(self.fully_quantized_mbv2.features_0_0_bn.weights_quantizers['weight'], WeightsPOTInferableQuantizer))

    def test_weights_activation_qc(self):
        self.assertTrue(len(self.fully_quantized_mbv2.classifier_1.weights_quantizers) == 1)
        self.assertTrue(isinstance(self.fully_quantized_mbv2.classifier_1.weights_quantizers['weight'], WeightsPOTInferableQuantizer))
        self.assertTrue(isinstance(self.fully_quantized_mbv2.classifier_1_activation_holder_quantizer.activation_holder_quantizer, ActivationPOTInferableQuantizer))

    def test_activation_qc(self):
        self.assertTrue(isinstance(self.fully_quantized_mbv2.features_0_2_activation_holder_quantizer.activation_holder_quantizer,
                                   ActivationPOTInferableQuantizer))

    def test_no_quantization_qc(self):
        self.assertTrue(isinstance(self.fully_quantized_mbv2.features_0_2, torch.nn.ReLU6))
        self.assertTrue(isinstance(self.fully_quantized_mbv2.classifier_0, torch.nn.Dropout))

    def test_save_and_load_model(self):
        float_model_filename = f'mbv2_float.pth'
        model_filename = f'mbv2_fq.pth'
        model_folder = '/tmp/'
        model_file = os.path.join(model_folder, model_filename)
        float_model_file = os.path.join(model_folder, float_model_filename)
        torch.save(self.fully_quantized_mbv2.state_dict(), model_file)
        torch.save(self.mbv2.state_dict(), float_model_file)
        print(f"Pytorch .pth Model: {model_file}")
        print(f"Float Pytorch .pth Model: {float_model_file}")

        model = copy.deepcopy(self.fully_quantized_mbv2)
        model.load_state_dict(torch.load(model_file))
        model.eval()
        model(next(self.representative_data_gen()))

        torch_traced = torch.jit.trace(self.fully_quantized_mbv2, to_torch_tensor(next(self.representative_data_gen())))
        torch_script_filename = f'mbv2_fq_torchscript.pth'
        torch_script_model = torch.jit.script(torch_traced)
        torch_script_file = os.path.join(model_folder, torch_script_filename)
        torch.jit.save(torch_script_model, torch_script_file)
        print(f"Pytorch torch script Model: {torch_script_file}")

        loaded_script_model = torch.jit.load(torch_script_file)
        loaded_script_model.eval()
        images = next(self.representative_data_gen())[0]
        diff = loaded_script_model(images) - model(images)
        sum_abs_error = np.sum(np.abs(self._to_numpy(diff)))
        self.assertTrue(sum_abs_error==0, f'Difference between loaded torch script to loaded torch model: {sum_abs_error}')
