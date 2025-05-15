# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
import model_compression_toolkit as mct
import torch
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchPreservingActivationQuantizationHolder

from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, Signedness
from mct_quantizers import QuantizationMethod
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema


def representative_data_gen(shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
    for _ in range(num_iter):
        yield [torch.randn(batch_size, *shape)] * num_inputs


def get_float_model():
    class BaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
            self.dropout = torch.nn.Dropout()
            self.flatten1 = torch.nn.Flatten()
            self.flatten2 = torch.nn.Flatten()
            self.fc1 = torch.nn.Linear(108, 216)
            self.fc2 = torch.nn.Linear(216, 432)

        def forward(self, x):
            x = self.conv(x)
            x = self.flatten1(x)
            x = self.fc1(x)
            x = self.flatten2(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    return BaseModel()


def get_tpc(insert_preserving):
    base_config = schema.OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO)

    default_config = schema.OpQuantizationConfig(
        default_weight_attr_config=AttributeQuantizationConfig(),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        supported_input_activation_n_bits=8,
        enable_activation_quantization=True,
        quantization_preserving=False,
        signedness=Signedness.AUTO)

    mixed_precision_cfg_list = [base_config]
    default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([default_config]))
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list),
                                                                           base_config=base_config)

    operator_set = []
    preserving_quantization_config = (default_configuration_options.clone_and_edit(enable_activation_quantization=False, quantization_preserving=True))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.DROPOUT, qc_options=preserving_quantization_config))
    operator_set.append(schema.OperatorsSet(name=schema.OperatorSetNames.FLATTEN, qc_options=preserving_quantization_config))
    conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
    fc = schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED, qc_options=mixed_precision_configuration_options)
    operator_set.extend([conv, fc])

    tpc = schema.TargetPlatformCapabilities(
        default_qco=default_configuration_options,
        operator_set=tuple(operator_set),
        insert_preserving_quantizers=insert_preserving)

    return tpc


def test_quantization_preserving_holder():
    """
    This test uses a TPC with insert_preserving_quantizer enabled and verifies that all preserving operations in the
    model are followed by a suitable activation quantization holder with the correct attributes.
    """
    float_model = get_float_model()
    target_platform_cap = get_tpc(insert_preserving=True)
    core_config = CoreConfig()
    
    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
        in_module=float_model,
        representative_data_gen=representative_data_gen,
        core_config=core_config,
        target_platform_capabilities=target_platform_cap
    )

    # check conv
    conv_activation_holder_quantizer = quantized_model.conv_activation_holder_quantizer
    assert isinstance(conv_activation_holder_quantizer, PytorchActivationQuantizationHolder)

    # check flatten1 (same conv)
    flatten1_activation_holder_quantizer = quantized_model.flatten1_activation_holder_quantizer
    assert isinstance(flatten1_activation_holder_quantizer, PytorchPreservingActivationQuantizationHolder)
    assert flatten1_activation_holder_quantizer.quantization_bypass == True
    assert flatten1_activation_holder_quantizer.activation_holder_quantizer.num_bits == conv_activation_holder_quantizer.activation_holder_quantizer.num_bits
    assert flatten1_activation_holder_quantizer.activation_holder_quantizer.signed == conv_activation_holder_quantizer.activation_holder_quantizer.signed
    assert flatten1_activation_holder_quantizer.activation_holder_quantizer.threshold_np == conv_activation_holder_quantizer.activation_holder_quantizer.threshold_np

    # check fc1
    fc1_activation_holder_quantizer = quantized_model.fc1_activation_holder_quantizer
    assert isinstance(fc1_activation_holder_quantizer, PytorchActivationQuantizationHolder)

    # check flatten2 (same fc1)
    flatten2_activation_holder_quantizer = quantized_model.flatten2_activation_holder_quantizer
    assert isinstance(flatten2_activation_holder_quantizer, PytorchPreservingActivationQuantizationHolder)
    assert flatten2_activation_holder_quantizer.quantization_bypass == True
    assert flatten2_activation_holder_quantizer.activation_holder_quantizer.num_bits == fc1_activation_holder_quantizer.activation_holder_quantizer.num_bits
    assert flatten2_activation_holder_quantizer.activation_holder_quantizer.signed == fc1_activation_holder_quantizer.activation_holder_quantizer.signed
    assert flatten2_activation_holder_quantizer.activation_holder_quantizer.threshold_np == fc1_activation_holder_quantizer.activation_holder_quantizer.threshold_np
    
    # check dropout (same fc1)
    dropout_activation_holder_quantizer = quantized_model.dropout_activation_holder_quantizer
    assert isinstance(dropout_activation_holder_quantizer, PytorchPreservingActivationQuantizationHolder)
    assert dropout_activation_holder_quantizer.quantization_bypass == True
    assert dropout_activation_holder_quantizer.activation_holder_quantizer.num_bits == fc1_activation_holder_quantizer.activation_holder_quantizer.num_bits
    assert dropout_activation_holder_quantizer.activation_holder_quantizer.signed == fc1_activation_holder_quantizer.activation_holder_quantizer.signed
    assert dropout_activation_holder_quantizer.activation_holder_quantizer.threshold_np == fc1_activation_holder_quantizer.activation_holder_quantizer.threshold_np

    # check fc2
    fc2_activation_holder_quantizer = quantized_model.fc2_activation_holder_quantizer
    assert isinstance(fc2_activation_holder_quantizer, PytorchActivationQuantizationHolder)


def test_no_quantization_preserving_holder():
    """
    This test uses a TPC with insert_preserving_quantizer disabled and verifies that none of the preserving operations
    in the model is connected to a preserving quantization holder.
    """
    float_model = get_float_model()
    target_platform_cap = get_tpc(insert_preserving=False)
    core_config = CoreConfig()

    quantized_model, _ = mct.ptq.pytorch_post_training_quantization(
        in_module=float_model,
        representative_data_gen=representative_data_gen,
        core_config=core_config,
        target_platform_capabilities=target_platform_cap
    )

    # Check that no PytorchPreservingActivationQuantizationHolder exists in the model
    for _, module in quantized_model.named_modules():
        assert not isinstance(module, PytorchPreservingActivationQuantizationHolder), \
            f"Found unexpected PytorchPreservingActivationQuantizationHolder: {module}"

    # check conv
    assert hasattr(quantized_model, 'conv_activation_holder_quantizer')
    assert isinstance(quantized_model.conv_activation_holder_quantizer, PytorchActivationQuantizationHolder)

    # check flatten1 (same conv)
    assert not hasattr(quantized_model, 'flatten1_activation_holder_quantizer')

    # check fc1
    assert hasattr(quantized_model, 'fc1_activation_holder_quantizer')
    assert isinstance(quantized_model.fc1_activation_holder_quantizer, PytorchActivationQuantizationHolder)

    # check flatten2 (same fc1)
    assert not hasattr(quantized_model, 'flatten2_activation_holder_quantizer')

    # check dropout (same fc1)
    assert not hasattr(quantized_model, 'dropout_activation_holder_quantizer')

    # check fc2
    assert hasattr(quantized_model, 'fc2_activation_holder_quantizer')
    assert isinstance(quantized_model.fc2_activation_holder_quantizer, PytorchActivationQuantizationHolder)
             