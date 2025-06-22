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

from torch import nn
import numpy as np

from model_compression_toolkit.core.common.hessian import HessianInfoService, HessianScoresRequest, HessianMode, \
    HessianScoresGranularity
from model_compression_toolkit.core.pytorch.data_util import data_gen_to_dataloader
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=(3, 3))
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        y = self.relu(x)
        return y


class MultipleActNodesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(3, 3, kernel_size=(3, 3))
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        y = self.conv2(x)
        return y


def representative_dataset():
    for _ in range(2):
        yield [np.random.randn(2, 3, 16, 16).astype(np.float32)]


class BaseHessianServiceTest(BasePytorchTest):

    def __init__(self, unit_test, model, compute_hessian=True, run_verification=True):
        super().__init__(unit_test)

        self.float_model = model()
        self.request = None
        self.num_scores = None
        self.num_nodes = None
        self.graph = None
        self.pytorch_impl = PytorchImplementation()
        self.run_verification = run_verification
        self.compute_hessian = compute_hessian
        self.attach2fw = AttachTpcToPytorch()

    def verify_hessian(self):
        self.unit_test.assertEqual(len(self.hessian), self.num_nodes, f"Expecting returned Hessian list to include "
                                                                      f"{self.num_nodes} list of approximation.")

        for hess in self.hessian.values():
            self.unit_test.assertEqual(hess.shape[0], self.num_scores,
                                       f"Expecting {self.num_scores} Hessian scores.")

    def run_test(self, seed=0):
        # This is just an internal assertion for the test setup
        assert (self.request is not None and self.num_scores is not None and self.num_nodes is not None
                and self.graph is not None), "Test parameters are not initialized."

        self.hessian_service = HessianInfoService(graph=self.graph,
                                                  fw_impl=self.pytorch_impl)

        self.unit_test.assertEqual(self.hessian_service.graph, self.graph)
        self.unit_test.assertEqual(self.hessian_service.fw_impl, self.pytorch_impl)

        if self.compute_hessian:
            self.hessian = self.hessian_service.fetch_hessian(self.request)

        if self.run_verification:
            self.verify_hessian()


class FetchActivationHessianTest(BaseHessianServiceTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, model=BasicModel)

        self.num_nodes = 1
        self.num_scores = 2

    def run_test(self, seed=0):
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.pytorch_impl,
                                                representative_dataset,
                                                generate_pytorch_tpc,
                                                attach2fw=self.attach2fw)
        dataloader = data_gen_to_dataloader(representative_dataset, batch_size=1)
        self.request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                            granularity=HessianScoresGranularity.PER_TENSOR,
                                            target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]],
                                            data_loader=dataloader,
                                            n_samples=self.num_scores)

        super().run_test()


class FetchWeightsHessianTest(BaseHessianServiceTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, model=BasicModel)

        self.num_nodes = 1
        self.num_scores = 2

    def run_test(self, seed=0):
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.pytorch_impl,
                                                representative_dataset,
                                                generate_pytorch_tpc,
                                                attach2fw=self.attach2fw)

        self.request = HessianScoresRequest(mode=HessianMode.WEIGHTS,
                                            granularity=HessianScoresGranularity.PER_OUTPUT_CHANNEL,
                                            target_nodes=[list(self.graph.get_topo_sorted_nodes())[1]],
                                            data_loader=data_gen_to_dataloader(representative_dataset, batch_size=1),
                                            n_samples=self.num_scores)

        super().run_test()


class FetchHessianNotEnoughSamplesThrowTest(BaseHessianServiceTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, model=BasicModel, compute_hessian=False, run_verification=False)

        self.num_nodes = 1
        self.num_scores = 5

    def run_test(self, seed=0):
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.pytorch_impl,
                                                representative_dataset,
                                                generate_pytorch_tpc,
                                                attach2fw=self.attach2fw)
        data_loader = data_gen_to_dataloader(representative_dataset, batch_size=2)
        self.request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                            granularity=HessianScoresGranularity.PER_TENSOR,
                                            target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]],
                                            data_loader=data_loader,
                                            n_samples=self.num_scores)

        super().run_test()

        with self.unit_test.assertRaises(Exception) as e:
            self.hessian_service.fetch_hessian(self.request)  # representative dataset produces 4 images total
        self.unit_test.assertTrue('not enough samples in the provided representative dataset' in str(e.exception))


class FetchHessianNotEnoughSamplesSmallBatchThrowTest(BaseHessianServiceTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, model=BasicModel, compute_hessian=False, run_verification=False)

        self.num_nodes = 1
        self.num_scores = 5

    def run_test(self, seed=0):
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.pytorch_impl,
                                                representative_dataset,
                                                generate_pytorch_tpc,
                                                attach2fw=self.attach2fw)

        data_loader = data_gen_to_dataloader(representative_dataset, batch_size=1)
        self.request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                            granularity=HessianScoresGranularity.PER_TENSOR,
                                            target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]],
                                            data_loader=data_loader,
                                            n_samples=self.num_scores)

        super().run_test()

        with self.unit_test.assertRaises(Exception) as e:
            self.hessian_service.fetch_hessian(self.request)  # representative dataset produces 4 images total

        self.unit_test.assertTrue('not enough samples in the provided representative dataset' in str(e.exception))


class FetchComputeBatchLargerThanReprBatchTest(BaseHessianServiceTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, model=BasicModel, compute_hessian=False, run_verification=False)

        self.num_nodes = 1
        self.num_scores = 3

    def run_test(self, seed=0):
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.pytorch_impl,
                                                representative_dataset,
                                                generate_pytorch_tpc,
                                                attach2fw=self.attach2fw)

        data_loader = data_gen_to_dataloader(representative_dataset, batch_size=3)
        self.request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                            granularity=HessianScoresGranularity.PER_TENSOR,
                                            target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]],
                                            data_loader=data_loader,
                                            n_samples=self.num_scores)

        super().run_test()
        self.hessian = self.hessian_service.fetch_hessian(self.request)  # representative batch size is 2
        super().verify_hessian()


class FetchHessianRequiredZeroTest(BaseHessianServiceTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, model=BasicModel)

        self.num_nodes = 1
        self.num_scores = 0

    def run_test(self, seed=0):
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.pytorch_impl,
                                                representative_dataset,
                                                generate_pytorch_tpc,
                                                attach2fw=AttachTpcToPytorch())
        data_loader = data_gen_to_dataloader(representative_dataset, batch_size=1)
        self.request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                            granularity=HessianScoresGranularity.PER_TENSOR,
                                            target_nodes=[list(self.graph.get_topo_sorted_nodes())[0]],
                                            data_loader=data_loader,
                                            n_samples=self.num_scores)

        super().run_test()


class FetchHessianMultipleNodesTest(BaseHessianServiceTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, model=MultipleActNodesModel)

        self.num_nodes = 2
        self.num_scores = 2

    def run_test(self, seed=0):
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.pytorch_impl,
                                                representative_dataset,
                                                generate_pytorch_tpc,
                                                attach2fw=self.attach2fw)

        nodes = list(self.graph.get_topo_sorted_nodes())
        data_loader = data_gen_to_dataloader(representative_dataset, batch_size=1)
        self.request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                            granularity=HessianScoresGranularity.PER_TENSOR,
                                            target_nodes=[nodes[0], nodes[2]],
                                            data_loader=data_loader,
                                            n_samples=self.num_scores)

        super().run_test()


class DoubleFetchHessianTest(BaseHessianServiceTest):
    def __init__(self, unit_test):
        super().__init__(unit_test, model=MultipleActNodesModel, compute_hessian=False, run_verification=False)
        self.num_nodes = 1
        self.num_scores = 2

    def run_test(self, seed=0):
        self.graph = prepare_graph_with_configs(self.float_model,
                                                self.pytorch_impl,
                                                representative_dataset,
                                                generate_pytorch_tpc,
                                                attach2fw=self.attach2fw)

        target_node = list(self.graph.get_topo_sorted_nodes())[0]
        data_loader = data_gen_to_dataloader(representative_dataset, batch_size=1)
        self.request = HessianScoresRequest(mode=HessianMode.ACTIVATION,
                                            granularity=HessianScoresGranularity.PER_TENSOR,
                                            target_nodes=[target_node],
                                            data_loader=data_loader,
                                            n_samples=self.num_scores)

        super().run_test()

        hessian = self.hessian_service.fetch_hessian(self.request)
        self.unit_test.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.unit_test.assertEqual(hessian[target_node.name].shape[0], 2, "Expecting 2 Hessian scores.")

        hessian = self.hessian_service.fetch_hessian(self.request)
        self.unit_test.assertEqual(len(hessian), 1, "Expecting returned Hessian list to include one list of "
                                          "approximation, for the single target node.")
        self.unit_test.assertEqual(hessian[target_node.name].shape[0], 2, "Expecting 2 Hessian scores.")
