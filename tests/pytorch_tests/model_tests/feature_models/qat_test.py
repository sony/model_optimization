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
import torch.nn as nn
from torch import Tensor

from model_compression_toolkit.core.pytorch.utils import get_working_device
from model_compression_toolkit.qat.pytorch.quantizer.base_pytorch_qat_quantizer import BasePytorchQATTrainableQuantizer
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.get_all_subclasses import get_all_subclasses
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizers import \
    BasePyTorchInferableQuantizer
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model, \
    generate_tp_model_with_activation_mp
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
import model_compression_toolkit as mct
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit import quantizers_infrastructure as qi
from model_compression_toolkit import MixedPrecisionQuantizationConfigV2
from tests.pytorch_tests.tpc_pytorch import get_mp_activation_pytorch_tpc_dict
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs


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

    def run_test(self, experimental_facade=False):
        self._gen_fixed_input()
        model_float = self.create_networks()
        _tpc = self.get_tpc()
        ptq_model, quantization_info = mct.pytorch_post_training_quantization_experimental(model_float,
                                                                                           self.representative_data_gen_experimental,
                                                                                           target_platform_capabilities=_tpc,
                                                                                           new_experimental_exporter=True)

        qat_ready_model, quantization_info = mct.pytorch_quantization_aware_training_init(model_float,
                                                                                          self.representative_data_gen_experimental,
                                                                                          target_platform_capabilities=_tpc)

        if self.test_loading:
            pass # TODO: need to save and load pytorch model

        if self.finalize:
            qat_finalized_model = mct.pytorch_quantization_aware_training_finalize(qat_ready_model)
        else:
            qat_finalized_model = None

        self.compare(ptq_model,
                     qat_ready_model,
                     qat_finalized_model,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

    def compare(self, ptq_model, qat_ready_model, qat_finalized_model, input_x=None, quantization_info=None):
        all_trainable_quantizers = get_all_subclasses(BasePytorchQATTrainableQuantizer)
        # check relevant layers are wrapped and correct quantizers were chosen
        for _, layer in qat_ready_model.named_children():
            self.unit_test.assertTrue(isinstance(layer, qi.PytorchQuantizationWrapper))
            if isinstance(layer.layer, nn.SiLU):
                q = [_q for _q in all_trainable_quantizers if _q.quantizer_type == mct.TrainingMethod.STE
                     and _q.quantization_target == QuantizationTarget.Activation
                     and self.activation_quantization_method in _q.quantization_method]
                self.unit_test.assertTrue(len(q) == 1)
                self.unit_test.assertTrue(isinstance(layer.activation_quantizers[0], q[0]))
            if isinstance(layer.layer, nn.Conv2d):
                q = [_q for _q in all_trainable_quantizers if _q.quantizer_type == mct.TrainingMethod.STE
                     and _q.quantization_target == QuantizationTarget.Weights
                     and self.weights_quantization_method in _q.quantization_method]
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
                self.unit_test.assertTrue(isinstance(layer, qi.PytorchQuantizationWrapper))
                if isinstance(layer.layer, nn.SiLU):
                    q = [_q for _q in all_inferable_quantizers if
                         _q.quantization_target == QuantizationTarget.Activation
                         and self.activation_quantization_method in _q.quantization_method]
                    self.unit_test.assertTrue(len(q) == 1)
                    self.unit_test.assertTrue(isinstance(layer.activation_quantizers[0], q[0]))
                if isinstance(layer.layer, nn.Conv2d):
                    q = [_q for _q in all_inferable_quantizers if
                         _q.quantization_target == QuantizationTarget.Weights
                         and self.weights_quantization_method in _q.quantization_method]
                    self.unit_test.assertTrue(len(q) == 1)
                    self.unit_test.assertTrue(isinstance(layer.weights_quantizers['weight'], q[0]))
            # check quantization didn't change when switching between PTQ model and QAT finalized model
            qat_finalized_output = qat_finalized_model(_in).cpu().detach().numpy()
            self.unit_test.assertTrue(np.isclose(np.linalg.norm(qat_finalized_output - qat_ready_output) / np.linalg.norm(qat_ready_output), 0, atol=1e-6))


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

    def run_test(self, experimental_facade=False):
        self._gen_fixed_input()
        model_float = self.create_networks()
        config = mct.CoreConfig(mixed_precision_config=MixedPrecisionQuantizationConfigV2())
        kpi = mct.KPI() # inf memory
        qat_ready_model, quantization_info = mct.pytorch_quantization_aware_training_init(model_float,
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
            if layer.is_weights_quantization:
                self.unit_test.assertTrue(len(layer.weights_quantizers['weight'].quantization_config.weights_bits_candidates) > 1)
            if layer.is_activation_quantization:
                self.unit_test.assertTrue(len(layer.activation_quantizers[0].quantization_config.activation_bits_candidates) > 1)


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

    def run_test(self, experimental_facade=False):
        self._gen_fixed_input()
        model_float = self.create_networks()
        config = mct.CoreConfig(mixed_precision_config=MixedPrecisionQuantizationConfigV2())
        kpi = mct.KPI(weights_memory=50, activation_memory=40)
        qat_ready_model, quantization_info = mct.pytorch_quantization_aware_training_init(model_float,
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
            if layer.is_weights_quantization:
                self.unit_test.assertTrue(len(layer.weights_quantizers['weight'].quantization_config.weights_bits_candidates) > 1)
            if layer.is_activation_quantization:
                self.unit_test.assertTrue(len(layer.activation_quantizers[0].quantization_config.activation_bits_candidates) > 1)