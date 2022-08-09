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
from typing import List

import model_compression_toolkit as mct
import numpy as np
import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU

from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
from model_compression_toolkit.core.pytorch.constants import KERNEL
from tests.common_tests.helpers.activation_mp_tp_model import generate_tp_model_with_activation_mp
from tests.pytorch_tests.tpc_pytorch import generate_activation_mp_tpc_pytorch
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest


def small_random_datagen():
    return [np.random.random((1, 8, 8, 3))]


def large_random_datagen():
    return [np.random.random((1, 224, 224, 3))]


def compute_output_size(output_shape):
    output_shapes = output_shape if isinstance(output_shape, List) else [output_shape]
    output_shapes = [s[1:] for s in output_shapes]
    return sum([np.prod([x for x in output_shape if x is not None]) for output_shape in output_shapes])


class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.conv1 = Conv2d(8, 8, 3)
        self.bn = BatchNorm2d(8)
        self.relu = ReLU()

    def forward(self, inp):
        size = inp.shape
        x = self.conv1(inp)
        x = self.bn(x)
        x = self.relu(x)
        return x, size

    def parameters_sum(self):
        return getattr(self.conv1, KERNEL).detach().numpy().flatten().shape[0]

    def max_tensor(self):
        _, l_shape = self(torch.from_numpy(small_random_datagen()[0]).float())
        return compute_output_size(l_shape)


class ComplexModel(torch.nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.conv1 = Conv2d(224, 224, 1)
        self.bn1 = BatchNorm2d(224)
        self.relu1 = ReLU()

        self.conv2 = Conv2d(224, 250, 3, padding='same')
        self.bn2 = BatchNorm2d(250)
        self.relu2 = ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        size = x.shape
        x = self.bn2(x)
        x = self.relu2(x)

        return x, size

    def parameters_sum(self):
        return getattr(self.conv1, KERNEL).detach().numpy().flatten().shape[0] + \
               getattr(self.conv2, KERNEL).detach().numpy().flatten().shape[0]

    def max_tensor(self):
        _, l_shape = self(torch.from_numpy(large_random_datagen()[0]).float())
        return compute_output_size(l_shape)


def prep_test(model, mp_bitwidth_candidates_list, random_datagen):
    base_config, mixed_precision_cfg_list = get_op_quantization_configs()
    base_config = base_config.clone_and_edit(weights_n_bits=mp_bitwidth_candidates_list[0][0],
                                             activation_n_bits=mp_bitwidth_candidates_list[0][1])
    tp_model = generate_tp_model_with_activation_mp(
        base_cfg=base_config,
        mp_bitwidth_candidates_list=mp_bitwidth_candidates_list)
    tpc = generate_activation_mp_tpc_pytorch(tp_model=tp_model, name="kpi_data_test")

    kpi_data = mct.pytorch_kpi_data(in_model=model,
                                    representative_data_gen=random_datagen,
                                    target_platform_capabilities=tpc)

    return kpi_data


class KPIDataBaseTestClass(BasePytorchTest):

    def verify_results(self, kpi, sum_parameters, max_tensor):
        self.unit_test.assertTrue(kpi.weights_memory == sum_parameters,
                                  f"Expects weights_memory to be {sum_parameters} "
                                  f"but result is {kpi.weights_memory}")
        self.unit_test.assertTrue(kpi.activation_memory == max_tensor,
                                  f"Expects activation_memory to be {max_tensor} "
                                  f"but result is {kpi.activation_memory}")


class TestKPIDataBasicAllBitwidth(KPIDataBaseTestClass):

    def run_test(self):
        model = BasicModel()
        sum_parameters = model.parameters_sum()
        max_tensor = model.max_tensor()

        mp_bitwidth_candidates_list = [(i, j) for i in [8, 4, 2] for j in [8, 4, 2]]

        kpi_data = prep_test(model, mp_bitwidth_candidates_list, small_random_datagen)

        # max should be 8-bit quantization
        self.verify_results(kpi_data, sum_parameters, max_tensor)


class TestKPIDataBasicPartialBitwidth(KPIDataBaseTestClass):

    def run_test(self):
        model = BasicModel()
        sum_parameters = model.parameters_sum()
        max_tensor = model.max_tensor()

        mp_bitwidth_candidates_list = [(i, j) for i in [4, 2] for j in [4, 2]]

        kpi_data = prep_test(model, mp_bitwidth_candidates_list, small_random_datagen)

        self.verify_results(kpi_data, sum_parameters, max_tensor)


class TestKPIDataComplesAllBitwidth(KPIDataBaseTestClass):

    def run_test(self):
        model = ComplexModel()
        sum_parameters = model.parameters_sum()
        max_tensor = model.max_tensor()

        mp_bitwidth_candidates_list = [(i, j) for i in [8, 4, 2] for j in [8, 4, 2]]

        kpi_data = prep_test(model, mp_bitwidth_candidates_list, large_random_datagen)

        self.verify_results(kpi_data, sum_parameters, max_tensor)



class TestKPIDataComplexPartialBitwidth(KPIDataBaseTestClass):

    def run_test(self):
        model = ComplexModel()
        sum_parameters = model.parameters_sum()
        max_tensor = model.max_tensor()

        mp_bitwidth_candidates_list = [(i, j) for i in [4, 2] for j in [4, 2]]

        kpi_data = prep_test(model, mp_bitwidth_candidates_list, large_random_datagen)

        self.verify_results(kpi_data, sum_parameters, max_tensor)
