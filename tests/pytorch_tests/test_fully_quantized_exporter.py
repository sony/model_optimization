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
import numpy as np
import random
import unittest
from torchvision.models import mobilenet_v2

import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.core.tpc_models.default_tpc.v3.tpc_pytorch import generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model

tp = mct.target_platform

import torch


class TestFullyQuantizedExporter(unittest.TestCase):

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def test_sanity(self):
        """
        Test that new fully quantized exporter model outputs the same as
        old exported model.
        """
        model = mobilenet_v2(pretrained=True)
        repr_dataset = lambda: to_torch_tensor([torch.ones(1, 3, 224, 224)])
        # seed = np.random.randint(0, 100, size=1)[0]

        tp_model = generate_test_tp_model({
                                           # 'enable_weights_quantization': False,
                                           # 'enable_activation_quantization': False,
            # 'activation_n_bits':32
                                           })
        tpc = generate_pytorch_tpc(tp_model=tp_model, name='test')


        for seed in range(100):
            self.set_seed(seed)
            core_config = mct.CoreConfig(n_iter=1)
            old_export_model, _ = mct.pytorch_post_training_quantization_experimental(
                in_module=model,
                representative_data_gen=repr_dataset,
                core_config=core_config,
            target_platform_capabilities=tpc
            )

            self.set_seed(seed)
            core_config = mct.CoreConfig(n_iter=1)
            new_export_model, _ = mct.pytorch_post_training_quantization_experimental(
                in_module=model,
                core_config=core_config,
                representative_data_gen=repr_dataset,
                target_platform_capabilities=tpc,
                new_experimental_exporter=True)

            def _to_numpy(t):
                return t.cpu().detach().numpy()

            images = repr_dataset()
            diff = new_export_model(images) - old_export_model(images)
            print(f'Max abs error: {np.max(np.abs(diff.cpu().detach().numpy()))}')
            w_delta = np.sum(np.abs(_to_numpy(old_export_model.features_0_0_bn.weight) - _to_numpy(new_export_model.features_0_0_bn.layer.weight)))
            assert w_delta==0
            self.assertTrue(np.sum(np.abs(diff.cpu().detach().numpy())) == 0)
