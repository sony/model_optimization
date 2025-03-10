# # Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# import copy
#
# import pytest
#
# import torch
# from torch import nn
#
# from model_compression_toolkit.core.common import BaseNode
# from model_compression_toolkit.core.common.graph.edge import EDGE_SOURCE_INDEX, EDGE_SINK_INDEX
# from model_compression_toolkit.target_platform_capabilities import TargetPlatformCapabilities, OperatorsSet, Fusing
# from model_compression_toolkit.core import QuantizationConfig
# from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
# from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
# from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
# from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OperatorSetNames
# from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import AttachTpcToPytorch
#
# def data_gen():
#     yield [torch.rand(1, 3, 5, 5)]
#
# @pytest.fixture
# def minimal_tpc_with_fusing(default_quant_cfg_options):
#     return TargetPlatformCapabilities(
#         default_qco=default_quant_cfg_options,
#         tpc_platform_type='test',
#         operator_set=[OperatorsSet(name=OperatorSetNames.CONV),
#                       OperatorsSet(name=OperatorSetNames.RELU),
#                       OperatorsSet(name=OperatorSetNames.FULLY_CONNECTED),
#                       OperatorsSet(name=OperatorSetNames.SOFTMAX)],
#         fusing_patterns=[Fusing(operator_groups=(OperatorsSet(name=OperatorSetNames.CONV),
#                                                  OperatorsSet(name=OperatorSetNames.RELU))),
#                          Fusing(operator_groups=(OperatorsSet(name=OperatorSetNames.FULLY_CONNECTED),
#                                                  OperatorsSet(name=OperatorSetNames.SOFTMAX)))]
#     )
#
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Conv2d(3, 3, kernel_size=(3, 3))
#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(in_features=27, out_features=10)
#         self.softmax = nn.Softmax()
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.flatten(x)
#         x = self.linear(x)
#         x = self.softmax(x)
#         return x
#
#
# @pytest.fixture
# def graph_with_fusion_metadata(minimal_tpc_with_fusing):
#     Model()(next(data_gen())[0])
#
#     fw_impl = PytorchImplementation()
#     fw_info = DEFAULT_PYTORCH_INFO
#     model = Model()
#     fqc = AttachTpcToPytorch().attach(minimal_tpc_with_fusing)
#
#     graph_with_fusion_metadata = graph_preparation_runner(model,
#                                                           data_gen,
#                                                           QuantizationConfig(),
#                                                           fw_info=fw_info,
#                                                           fw_impl=fw_impl,
#                                                           fqc=fqc,
#                                                           mixed_precision_enable=False,
#                                                           running_gptq=False)
#     return graph_with_fusion_metadata
#
# #
# # def test_fail_validate_after_node_removal(graph_with_fusion_metadata):
# #     # Find the ReLU node by name
# #     relu_node = graph_with_fusion_metadata.find_node_by_name('relu')[0]
# #
# #     # Create a new node (Tanh) with compatible shapes
# #     new_node = BaseNode(
# #         name='tanh',
# #         framework_attr={},
# #         input_shape=relu_node.input_shape,
# #         output_shape=relu_node.output_shape,
# #         weights={},
# #         layer_class=nn.Tanh
# #     )
# #
# #     # Replace the ReLU node with the new Tanh node
# #     # Expect validation to fail since the original ReLU node is no longer in the graph
# #     with pytest.raises(ValueError):
# #         graph_with_fusion_metadata.replace_node(relu_node, new_node)
# #
# #
# # def test_fail_validate_after_topology_change(graph_with_fusion_metadata):
# #     # Find the Conv2d and Flatten nodes by name
# #     conv_node = graph_with_fusion_metadata.find_node_by_name('conv')[0]
# #     flatten_node = graph_with_fusion_metadata.find_node_by_name('flatten')[0]
# #
# #     # Add an edge from Conv2d to Flatten, breaking the linear chain
# #     # Expect validation to fail due to multiple successors
# #     with pytest.raises(ValueError):
# #         graph_with_fusion_metadata.add_edge(conv_node,
# #                                             flatten_node,
# #                                             **{EDGE_SOURCE_INDEX: 1, EDGE_SINK_INDEX: 1})
# #
# #
# # def test_fail_validate_after_adding_node_between_conv_to_relu(graph_with_fusion_metadata):
# #     # Find the Conv2d and ReLU nodes by name
# #     conv_node = graph_with_fusion_metadata.find_node_by_name('conv')[0]
# #     relu_node = graph_with_fusion_metadata.find_node_by_name('relu')[0]
# #
# #     # Remove the edge between Conv2d and ReLU
# #     # Expect validation to fail as the fused sequence is broken
# #     with pytest.raises(ValueError):
# #         graph_with_fusion_metadata.remove_edge(conv_node, relu_node)
# #
# #
# # def test_fail_validate_after_modifying_node_inputs(graph_with_fusion_metadata):
# #     new_input_node = BaseNode(
# #         name='new_input',
# #         framework_attr={},
# #         input_shape=None,
# #         output_shape=None,
# #         weights={},
# #         layer_class=nn.Identity
# #     )
# #     relu_node = graph_with_fusion_metadata.find_node_by_name('relu')[0]
# #     graph_with_fusion_metadata.add_node(new_input_node)
# #
# #     # Validation should fail due to extra input
# #     with pytest.raises(ValueError):
# #         graph_with_fusion_metadata.add_edge(new_input_node, relu_node, **{EDGE_SOURCE_INDEX: 1, EDGE_SINK_INDEX: 1})
# #
# #
# # def test_fail_validate_after_serialization_deserialization(graph_with_fusion_metadata):
# #     # Serialize and deserialize
# #     graph_copy = copy.deepcopy(graph_with_fusion_metadata)
# #     graph_copy.validate()
# #
# #     # Break fusing by removing an edge
# #     conv_node = graph_copy.find_node_by_name('conv')[0]
# #     relu_node = graph_copy.find_node_by_name('relu')[0]
# #
# #     # Validation should fail
# #     with pytest.raises(ValueError):
# #         graph_copy.remove_edge(conv_node, relu_node)
# #
# #
# #
#
#
#
#
