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
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch import Tensor

import model_compression_toolkit as mct
from mct_quantizers import PytorchActivationQuantizationHolder, QuantizationTarget, PytorchQuantizationWrapper
from mct_quantizers.common.get_all_subclasses import get_all_subclasses
from mct_quantizers.pytorch.quantizers import BasePyTorchInferableQuantizer
from model_compression_toolkit.core import MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.pytorch.utils import get_working_device, to_torch_tensor
from model_compression_toolkit.qat.pytorch.quantizer.base_pytorch_qat_quantizer import BasePytorchQATTrainableQuantizer
from model_compression_toolkit.qat.pytorch.quantizer.ste_rounding.symmetric_ste import STEActivationQATQuantizer
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
    get_op_quantization_configs
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model, \
    generate_tp_model_with_activation_mp
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from tests.pytorch_tests.tpc_pytorch import get_mp_activation_pytorch_tpc_dict
from mct_quantizers.common.base_inferable_quantizer import QuantizerID


def dummy_train(qat_ready_model, x, y):

    # Create a DataLoader for the dataset
    dataset = data.TensorDataset(x, y)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize the model, loss function, and optimizer
    # model = SimpleNN(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(qat_ready_model.parameters(), lr=0.0)

    # Training loop
    for epoch in range(1):
        for batch_x, batch_y in dataloader:
            # Forward pass
            outputs = qat_ready_model(batch_x.to(get_working_device()))
            loss = criterion(outputs, batch_y.to(get_working_device()))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return qat_ready_model

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1))
        self.activation = nn.SiLU()

    def forward(self, inp):
        x0 = self.conv1(inp)
        x1 = self.activation(x0)
        x2 = self.conv2(x1)
        y = self.activation(x2)
        return y

def repr_datagen():
    for _ in range(10):
        yield [np.random.random((4, 3, 224, 224))]


class QuantizationAwareTrainingTest(BasePytorchFeatureNetworkTest):
    def __init__(self, unit_test, weight_bits=2, activation_bits=4,
                 weights_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                 activation_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                 finalize=False, test_loading=False):

        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.finalize = finalize
        self.weights_quantization_method = weights_quantization_method
        self.activation_quantization_method = activation_quantization_method
        self.test_loading = test_loading
        super().__init__(unit_test, input_shape=(3, 4, 4))

    def get_tpc(self):
        return generate_pytorch_tpc(
            name="qat_test",
            tp_model=generate_test_tp_model({'weights_n_bits': self.weight_bits,
                                             'activation_n_bits': self.activation_bits,
                                             'weights_quantization_method': self.weights_quantization_method,
                                             'activation_quantization_method': self.activation_quantization_method}))

    def create_networks(self):
        return TestModel()

    def _gen_fixed_input(self):
        self.fixed_inputs = self.generate_inputs()

    def representative_data_gen_experimental(self):
        for _ in range(self.num_calibration_iter):
            yield self.fixed_inputs

    def run_test(self):
        self._gen_fixed_input()
        model_float = self.create_networks()
        _tpc = self.get_tpc()
        ptq_model, quantization_info = mct.ptq.pytorch_post_training_quantization_experimental(model_float,
                                                                                           self.representative_data_gen_experimental,
                                                                                           target_platform_capabilities=_tpc,
                                                                                           new_experimental_exporter=True)


        qat_ready_model, quantization_info = mct.qat.pytorch_quantization_aware_training_init(model_float,
                                                                                          self.representative_data_gen_experimental,
                                                                                          target_platform_capabilities=_tpc)
        copy_of_qat_ready_model = copy.deepcopy(qat_ready_model)

        # Generate a random dataset
        x = to_torch_tensor(next(self.representative_data_gen_experimental())[0])
        y = torch.rand(list(qat_ready_model(x).shape))

        # Train the model for one epoch with LR 0 and assert predictions identical before and after
        a = qat_ready_model(x)
        qat_ready_model = dummy_train(qat_ready_model,x, y)
        b = qat_ready_model(x)
        self.unit_test.assertTrue(torch.max(torch.abs(a - b)) == 0,
                                  f'QAT ready model was trained using LR 0 thus predictions should '
                                  f'be identical but a diff observed {torch.max(torch.abs(a - b))}')

        if self.test_loading:
            _path = tempfile.mkstemp('.pt')[1]
            torch.save(qat_ready_model, _path)
            qat_ready_model = torch.load(_path)

        if self.finalize:
            qat_finalized_model = mct.qat.pytorch_quantization_aware_training_finalize(qat_ready_model)
        else:
            qat_finalized_model = None

        self.compare(ptq_model,
                     copy_of_qat_ready_model,
                     qat_finalized_model,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

    def compare(self, ptq_model, qat_ready_model, qat_finalized_model, input_x=None, quantization_info=None):
        all_trainable_quantizers = get_all_subclasses(BasePytorchQATTrainableQuantizer)
        # check relevant layers are wrapped and correct quantizers were chosen
        for _, layer in qat_ready_model.named_children():
            if isinstance(layer, PytorchActivationQuantizationHolder):
                q = [_q for _q in all_trainable_quantizers if _q.identifier == mct.qat.TrainingMethod.STE
                     and _q.quantization_target == QuantizationTarget.Activation
                     and self.activation_quantization_method in _q.quantization_method]
                self.unit_test.assertTrue(len(q) == 1)
                self.unit_test.assertTrue(isinstance(layer.activation_holder_quantizer, q[0]))
            elif isinstance(layer, PytorchQuantizationWrapper) and isinstance(layer.layer, nn.Conv2d):
                q = [_q for _q in all_trainable_quantizers if _q.identifier == mct.qat.TrainingMethod.STE
                     and _q.quantization_target == QuantizationTarget.Weights
                     and self.weights_quantization_method in _q.quantization_method
                     and type(_q.identifier) == mct.qat.TrainingMethod]
                self.unit_test.assertTrue(len(q) == 1)
                self.unit_test.assertTrue(isinstance(layer.weights_quantizers['weight'], q[0]))

        # check quantization didn't change when switching between PTQ model and QAT ready model
        _in = Tensor(input_x[0]).to(get_working_device())
        ptq_output = ptq_model(_in).cpu().detach().numpy()
        qat_ready_output = qat_ready_model(_in).cpu().detach().numpy()
        self.unit_test.assertTrue(np.isclose(np.linalg.norm(ptq_output - qat_ready_output) / np.linalg.norm(ptq_output), 0, atol=1e-6))
        if self.finalize:
            all_inferable_quantizers = get_all_subclasses(BasePyTorchInferableQuantizer)
            for _, layer in qat_finalized_model.named_children():
                if isinstance(layer, PytorchActivationQuantizationHolder):
                    q = [_q for _q in all_inferable_quantizers if
                         _q.quantization_target == QuantizationTarget.Activation
                         and self.activation_quantization_method in _q.quantization_method
                         and _q.identifier == QuantizerID.INFERABLE]
                    self.unit_test.assertTrue(len(q) == 1)
                    self.unit_test.assertTrue(isinstance(layer.activation_holder_quantizer, q[0]))
                elif isinstance(layer, PytorchQuantizationWrapper) and isinstance(layer.layer, nn.Conv2d):
                    q = [_q for _q in all_inferable_quantizers if
                         _q.quantization_target == QuantizationTarget.Weights
                         and self.weights_quantization_method in _q.quantization_method
                         and _q.identifier == QuantizerID.INFERABLE]
                    self.unit_test.assertTrue(len(q) == 1)
                    self.unit_test.assertTrue(isinstance(layer.weights_quantizers['weight'], q[0]))
            # check quantization didn't change when switching between PTQ model and QAT finalized model
            qat_finalized_output = qat_finalized_model(_in).cpu().detach().numpy()
            self.unit_test.assertTrue(np.isclose(np.linalg.norm(qat_finalized_output - qat_ready_output) / np.linalg.norm(qat_ready_output), 0, atol=1e-6))


class QuantizationAwareTrainingQuantizerHolderTest(QuantizationAwareTrainingTest):
    def __init__(self, unit_test, finalize=False):
        super().__init__(unit_test, finalize=finalize)

    def get_tpc(self):
        return generate_pytorch_tpc(
            name="qat_test",
            tp_model=generate_test_tp_model({'weights_n_bits': self.weight_bits,
                                             'activation_n_bits': self.activation_bits,
                                             'weights_quantization_method': self.weights_quantization_method,
                                             'activation_quantization_method': self.activation_quantization_method}))

    def create_networks(self):
        return TestModel()


    def compare(self, ptq_model, qat_ready_model, qat_finalized_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(isinstance(qat_ready_model.inp_activation_holder_quantizer, PytorchActivationQuantizationHolder))
        self.unit_test.assertTrue(isinstance(qat_ready_model.activation_activation_holder_quantizer, PytorchActivationQuantizationHolder))
        self.unit_test.assertTrue(isinstance(qat_ready_model.activation_1_activation_holder_quantizer, PytorchActivationQuantizationHolder))
        self.unit_test.assertTrue(qat_ready_model.inp_activation_holder_quantizer.activation_holder_quantizer.quantization_config.activation_n_bits == 4)
        self.unit_test.assertTrue(qat_ready_model.activation_activation_holder_quantizer.activation_holder_quantizer.quantization_config.activation_n_bits == 4)
        self.unit_test.assertTrue(qat_ready_model.activation_1_activation_holder_quantizer.activation_holder_quantizer.quantization_config.activation_n_bits == 4)
        self.unit_test.assertTrue(isinstance(qat_ready_model.inp_activation_holder_quantizer.activation_holder_quantizer, STEActivationQATQuantizer))
        self.unit_test.assertTrue(isinstance(qat_ready_model.activation_activation_holder_quantizer.activation_holder_quantizer, STEActivationQATQuantizer))
        self.unit_test.assertTrue(isinstance(qat_ready_model.activation_1_activation_holder_quantizer.activation_holder_quantizer, STEActivationQATQuantizer))

        # Assert any other weights that added to quantization holder layer (only one allowed)
        self.unit_test.assertTrue(len(qat_ready_model.inp_activation_holder_quantizer.state_dict()) == 1)
        self.unit_test.assertTrue(len(qat_ready_model.activation_activation_holder_quantizer.state_dict()) == 1)
        self.unit_test.assertTrue(len(qat_ready_model.activation_1_activation_holder_quantizer.state_dict()) == 1)


class QuantizationAwareTrainingMixedPrecisionCfgTest(QuantizationAwareTrainingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        return get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tp_model_with_activation_mp(
                base_cfg=base_config,
                mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                             (4, 8), (4, 4), (4, 2),
                                             (2, 8), (2, 4), (2, 2)]),
            test_name='qat_test',
            tpc_name='qat_test_tpc')['qat_test']

    def run_test(self):
        self._gen_fixed_input()
        model_float = self.create_networks()
        config = mct.core.CoreConfig(mixed_precision_config=MixedPrecisionQuantizationConfigV2())
        kpi = mct.core.KPI() # inf memory
        qat_ready_model, quantization_info = mct.qat.pytorch_quantization_aware_training_init(model_float,
                                                                                          self.representative_data_gen_experimental,
                                                                                          kpi,
                                                                                          core_config=config,
                                                                                          fw_info=self.get_fw_info(),
                                                                                          target_platform_capabilities=self.get_tpc())

        self.compare(qat_ready_model,
                     qat_ready_model,
                     qat_ready_model,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

        # check that MP search returns 8 bits configuration for all layers
        self.unit_test.assertTrue(all(quantization_info.mixed_precision_cfg == [0, 0, 0, 0, 0]))

        # check that quantizer gets multiple bits configuration
        for _, layer in qat_ready_model.named_children():
            if isinstance(layer, PytorchActivationQuantizationHolder):
                self.unit_test.assertTrue(len(layer.activation_holder_quantizer.quantization_config.activation_bits_candidates) > 1)
            elif isinstance(layer, PytorchQuantizationWrapper) and layer.is_weights_quantization:
                self.unit_test.assertTrue(len(layer.weights_quantizers['weight'].quantization_config.weights_bits_candidates) > 1)



class QuantizationAwareTrainingMixedPrecisionKpiCfgTest(QuantizationAwareTrainingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        base_config, _ = get_op_quantization_configs()
        return get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tp_model_with_activation_mp(
                base_cfg=base_config,
                mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                             (4, 8), (4, 4), (4, 2),
                                             (2, 8), (2, 4), (2, 2)]),
            test_name='qat_test',
            tpc_name='qat_test_tpc')['qat_test']

    def run_test(self):
        self._gen_fixed_input()
        model_float = self.create_networks()
        config = mct.core.CoreConfig(mixed_precision_config=MixedPrecisionQuantizationConfigV2())
        kpi = mct.core.KPI(weights_memory=50, activation_memory=40)
        qat_ready_model, quantization_info = mct.qat.pytorch_quantization_aware_training_init(model_float,
                                                                                          self.representative_data_gen_experimental,
                                                                                          kpi,
                                                                                          core_config=config,
                                                                                          fw_info=self.get_fw_info(),
                                                                                          target_platform_capabilities=self.get_tpc())

        self.compare(qat_ready_model,
                     qat_ready_model,
                     qat_ready_model,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

        # check that MP search doesn't return 8 bits configuration for all layers
        self.unit_test.assertTrue(all(quantization_info.mixed_precision_cfg == [1, 1, 0, 0, 0]))

        # check that quantizer gets multiple bits configuration
        for _, layer in qat_ready_model.named_children():
            if isinstance(layer, PytorchActivationQuantizationHolder):
                self.unit_test.assertTrue(len(layer.activation_holder_quantizer.quantization_config.activation_bits_candidates) > 1)
            elif isinstance(layer, PytorchQuantizationWrapper) and layer.is_weights_quantization:
                self.unit_test.assertTrue(len(layer.weights_quantizers['weight'].quantization_config.weights_bits_candidates) > 1)