import os, inspect
import pytest
import torch
import onnx
import onnxruntime as ort
import numpy as np

import mct_quantizers as mctq
from model_compression_toolkit.ptq.pytorch.quantization_facade import pytorch_post_training_quantization
from model_compression_toolkit.qat.pytorch.quantization_facade import pytorch_quantization_aware_training_init_experimental, \
    pytorch_quantization_aware_training_finalize_experimental
from model_compression_toolkit.exporter import PytorchExportSerializationFormat, pytorch_export_model, QuantizationFormat
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, \
    QuantizationConfigOptions, OpQuantizationConfig, Signedness, TargetPlatformCapabilities


def rmse(x, y):
    return np.sqrt(np.power(x - y, 2).mean())


def onnx_reader(model_file, quantization_holder):
    model = onnx.load(model_file)

    constants = {}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            if node.output:
                tensor = None
                for attr in node.attribute:
                    if attr.name == 'value':
                        tensor = onnx.numpy_helper.to_array(attr.t)
                        break
                if tensor is not None:
                    constants[node.output[0]] = tensor

    model_dict = {}
    for node in model.graph.node:
        model_dict[node.name] = {'name': node.name,
                                 'op': node.op_type,
                                 'attributes': {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
                                 }
        if node.op_type == 'Gemm':
            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    model_dict[node.name]['attributes']['weight_value'] = onnx.numpy_helper.to_array(init)
                elif init.name == node.input[2]:
                    model_dict[node.name]['attributes']['bias_value'] = onnx.numpy_helper.to_array(init)
        elif 'Weights' in node.op_type and 'Quantizer' in node.op_type:
            if len(node.input) == 3:
                model_dict[node.name]['attributes']['min_value'] = constants.get(node.input[1])
                model_dict[node.name]['attributes']['max_value'] = constants.get(node.input[2])
            elif len(node.input) == 2:
                model_dict[node.name]['attributes']['threshold'] = constants.get(node.input[1])
        elif node.op_type == 'QuantizeLinear':
            # Get scale (input[1]) and zero_point (input[2]) if available
            scale_name = node.input[1] if len(node.input) > 1 else None
            zero_point_name = node.input[2] if len(node.input) > 2 else None

            # Find the actual values from initializers
            scale_value = None
            zero_point_value = None

            for init in model.graph.initializer:
                if init.name == scale_name:
                    scale_value = onnx.numpy_helper.to_array(init)
                elif init.name == zero_point_name:
                    zero_point_value = onnx.numpy_helper.to_array(init)

            if scale_value is None:
                scale_value = constants.get(scale_name)
            assert scale_value is not None

            if zero_point_value is None:
                zero_point_value = constants.get(zero_point_name)
            assert zero_point_value is not None

            if type(quantization_holder).__name__ == 'ActivationUniformInferableQuantizer':
                model_dict[node.name]['attributes']['min_value'] = scale_value
                model_dict[node.name]['attributes']['max_value'] = zero_point_value
            else:
                model_dict[node.name]['attributes']['threshold'] = scale_value

    return model_dict


def onnx_runner(model_file, model_input, is_mctq=False):
    # Load the model
    if is_mctq:
        session = ort.InferenceSession(model_file,
                                       mctq.get_ort_session_options(),
                                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    else:
        session = ort.InferenceSession(model_file)

    # Run inference
    input_feed = {i.name: m_input for i, m_input in zip(session.get_inputs(), model_input)}
    outputs = session.run(None, input_feed)

    # Get the result
    return outputs


def get_tpc(wbits=4, abits=4, a_qmethod=mctq.QuantizationMethod.POWER_OF_TWO, w_qmethod=mctq.QuantizationMethod.POWER_OF_TWO):
    default_weight_attr_config = AttributeQuantizationConfig(
        weights_quantization_method=w_qmethod,
        weights_n_bits=wbits,
        weights_per_channel_threshold=False,
        enable_weights_quantization=True,
        lut_values_bitwidth=None)

    op_cfg = OpQuantizationConfig(
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={},
        activation_quantization_method=a_qmethod,
        activation_n_bits=abits,
        supported_input_activation_n_bits=[abits],
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=Signedness.AUTO)

    cfg_options = QuantizationConfigOptions(quantization_configurations=[op_cfg])

    return TargetPlatformCapabilities(default_qco=cfg_options,
                                      tpc_platform_type='export_test',
                                      operator_set=None,
                                      fusing_patterns=None)


class ExportModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.linear(x)


class TestExporter:
    def setup_method(self):
        self.in_channels = 3
        self.out_channels = 4
        self.onnx_file = f"./tmp_model_{np.random.randint(1e10)}.onnx"
        self.qname_dict = {mctq.QuantizationMethod.POWER_OF_TWO: 'ActivationPOTQuantizer',
                           mctq.QuantizationMethod.SYMMETRIC: 'ActivationSymmetricQuantizer',
                           mctq.QuantizationMethod.UNIFORM: 'ActivationUniformQuantizer'
                           }
        self.wqname_dict = {mctq.QuantizationMethod.POWER_OF_TWO: 'WeightsPOTQuantizer',
                            mctq.QuantizationMethod.SYMMETRIC: 'WeightsSymmetricQuantizer',
                            mctq.QuantizationMethod.UNIFORM: 'WeightsUniformQuantizer'
                            }

    def teardown_method(self):
        if os.path.exists(self.onnx_file):
            os.remove(self.onnx_file)

    def get_model(self):
        return ExportModel(self.in_channels, self.out_channels)

    def representative_dataset(self, num_inputs):
        def rep_dataset():
            yield [np.random.randn(512, self.in_channels) for _ in range(num_inputs)]
        return rep_dataset

    def _run_mct(self, float_model, rep_dataset, abits, a_qmethod, w_qmethod=mctq.QuantizationMethod.POWER_OF_TWO):
        quantized_model, _ = pytorch_post_training_quantization(
            float_model,
            rep_dataset,
            target_platform_capabilities=get_tpc(abits=abits, a_qmethod=a_qmethod, w_qmethod=w_qmethod)
        )
        return quantized_model

    def _run_mct_qat(self, float_model, rep_dataset, abits, a_qmethod):
        qat_ready_model, _ = pytorch_quantization_aware_training_init_experimental(
            float_model,
            rep_dataset,
            target_platform_capabilities=get_tpc(abits=abits, a_qmethod=a_qmethod)
        )
        quantized_model = pytorch_quantization_aware_training_finalize_experimental(qat_ready_model)
        return quantized_model

    def _run_exporter(self, quantized_model, rep_dataset, quantization_format):
        pytorch_export_model(quantized_model,
                             save_model_path=self.onnx_file,
                             repr_dataset=rep_dataset,
                             serialization_format=PytorchExportSerializationFormat.ONNX,
                             quantization_format=quantization_format)

        return onnx_reader(self.onnx_file, quantized_model.linear_activation_holder_quantizer.activation_holder_quantizer)

    def _assert_outputs_match(self, quantized_model, rep_dataset, quantization_format, tol=1e-8):
        model_input = [i.astype(np.float32) for i in next(rep_dataset())]
        onnx_outputs = onnx_runner(self.onnx_file, model_input,
                                   is_mctq=quantization_format == QuantizationFormat.MCTQ)
        torch_outputs = quantized_model(*model_input)
        if not isinstance(torch_outputs, (list, tuple)):
            torch_outputs = [torch_outputs]
        torch_outputs = [o.detach().cpu().numpy() for o in torch_outputs]

        assert np.all([np.isclose(rmse(onnx_output, torch_output), 0, atol=tol)
                       for onnx_output, torch_output in zip(onnx_outputs, torch_outputs)])

    def _assert_quant_params_match(self, quantized_model, onnx_model_dict, a_qmethod, w_qmethod=mctq.QuantizationMethod.POWER_OF_TWO):
        assert quantized_model.x_activation_holder_quantizer.activation_holder_quantizer.num_bits == \
               onnx_model_dict[f'/x_activation_holder_quantizer/{self.qname_dict[a_qmethod]}']['attributes']['num_bits']
        if a_qmethod == mctq.QuantizationMethod.UNIFORM:
            assert np.isclose(quantized_model.x_activation_holder_quantizer.activation_holder_quantizer.min_range,
                              onnx_model_dict[f'/x_activation_holder_quantizer/{self.qname_dict[a_qmethod]}']['attributes']['min_range'])
            assert np.isclose(quantized_model.x_activation_holder_quantizer.activation_holder_quantizer.max_range,
                              onnx_model_dict[f'/x_activation_holder_quantizer/{self.qname_dict[a_qmethod]}']['attributes']['max_range'])
        else:
            assert np.isclose(quantized_model.x_activation_holder_quantizer.activation_holder_quantizer.threshold_np,
                              onnx_model_dict[f'/x_activation_holder_quantizer/{self.qname_dict[a_qmethod]}']['attributes']['threshold'])

        assert quantized_model.linear.weights_quantizers['weight'].num_bits == \
               onnx_model_dict[f'/linear/{self.wqname_dict[w_qmethod]}']['attributes']['num_bits']
        if w_qmethod == mctq.QuantizationMethod.UNIFORM:
            assert np.isclose(quantized_model.linear.weights_quantizers['weight'].adjusted_min_range_np,
                              onnx_model_dict[f'/linear/{self.wqname_dict[w_qmethod]}']['attributes']['min_value'])
            assert np.isclose(quantized_model.linear.weights_quantizers['weight'].adjusted_max_range_np,
                              onnx_model_dict[f'/linear/{self.wqname_dict[w_qmethod]}']['attributes']['max_value'])
        else:
            assert np.isclose(quantized_model.linear.weights_quantizers['weight'].threshold_np,
                              onnx_model_dict[f'/linear/{self.wqname_dict[w_qmethod]}']['attributes']['threshold'])

        assert quantized_model.linear_activation_holder_quantizer.activation_holder_quantizer.num_bits == \
               onnx_model_dict[f'/linear_activation_holder_quantizer/{self.qname_dict[a_qmethod]}']['attributes']['num_bits']
        if a_qmethod == mctq.QuantizationMethod.UNIFORM:
            assert np.isclose(quantized_model.linear_activation_holder_quantizer.activation_holder_quantizer.min_range,
                              onnx_model_dict[f'/linear_activation_holder_quantizer/{self.qname_dict[a_qmethod]}']['attributes']['min_range'])
            assert np.isclose(quantized_model.linear_activation_holder_quantizer.activation_holder_quantizer.max_range,
                              onnx_model_dict[f'/linear_activation_holder_quantizer/{self.qname_dict[a_qmethod]}']['attributes']['max_range'])
        else:
            assert np.isclose(quantized_model.linear_activation_holder_quantizer.activation_holder_quantizer.threshold_np,
                              onnx_model_dict[f'/linear_activation_holder_quantizer/{self.qname_dict[a_qmethod]}']['attributes']['threshold'])

    def _assert_fq_quant_params_match(self, quantized_model, onnx_model_dict, a_qmethod):
        if quantized_model.x_activation_holder_quantizer.activation_holder_quantizer.num_bits == 8:
            if a_qmethod == mctq.QuantizationMethod.UNIFORM:
                assert np.isclose(quantized_model.x_activation_holder_quantizer.activation_holder_quantizer.min_range,
                                  onnx_model_dict[f'/x_activation_holder_quantizer/QuantizeLinear']['attributes']['min_range'])
                assert np.isclose(quantized_model.x_activation_holder_quantizer.activation_holder_quantizer.max_range,
                                  onnx_model_dict[f'/x_activation_holder_quantizer/QuantizeLinear']['attributes']['max_range'])
            else:
                assert np.isclose(quantized_model.x_activation_holder_quantizer.activation_holder_quantizer.threshold_np,
                                  onnx_model_dict[f'/x_activation_holder_quantizer/QuantizeLinear']['attributes']['threshold'] *
                                  256 / (1+quantized_model.x_activation_holder_quantizer.activation_holder_quantizer.signed))
        assert np.all(quantized_model.linear.get_quantized_weights()['weight'].detach().cpu().numpy() ==
                      onnx_model_dict['/linear/layer/Gemm']['attributes']['weight_value'])
        if quantized_model.linear_activation_holder_quantizer.activation_holder_quantizer.num_bits == 8:
            if a_qmethod == mctq.QuantizationMethod.UNIFORM:
                assert np.isclose(quantized_model.linear_activation_holder_quantizer.activation_holder_quantizer.min_range,
                                  onnx_model_dict[f'/linear_activation_holder_quantizer/QuantizeLinear']['attributes']['min_range'])
                assert np.isclose(quantized_model.linear_activation_holder_quantizer.activation_holder_quantizer.max_range,
                                  onnx_model_dict[f'/linear_activation_holder_quantizer/QuantizeLinear']['attributes']['max_range'])
            else:
                assert np.isclose(quantized_model.linear_activation_holder_quantizer.activation_holder_quantizer.threshold_np,
                                  onnx_model_dict[f'/linear_activation_holder_quantizer/QuantizeLinear']['attributes']['threshold'] *
                                  256 / (1+quantized_model.linear_activation_holder_quantizer.activation_holder_quantizer.signed))

    @pytest.mark.parametrize('w_qmethod', [mctq.QuantizationMethod.POWER_OF_TWO,
                                           mctq.QuantizationMethod.SYMMETRIC,
                                           mctq.QuantizationMethod.UNIFORM])
    @pytest.mark.parametrize('a_qmethod, tol', [(mctq.QuantizationMethod.POWER_OF_TWO, 1e-8),
                                                (mctq.QuantizationMethod.SYMMETRIC, 1e-2),
                                                (mctq.QuantizationMethod.UNIFORM, 1e-2)])
    @pytest.mark.parametrize('abits', [8, 16])
    def test_mct_ptq_and_exporter_mctq(self, w_qmethod, abits, a_qmethod, tol):
        quantized_model = self._run_mct(self.get_model(), self.representative_dataset(1), abits, a_qmethod, w_qmethod)
        onnx_model_dict = self._run_exporter(quantized_model, self.representative_dataset(1), QuantizationFormat.MCTQ)

        self._assert_quant_params_match(quantized_model, onnx_model_dict, a_qmethod, w_qmethod)
        self._assert_outputs_match(quantized_model, self.representative_dataset(1), QuantizationFormat.MCTQ, tol=tol)

    @pytest.mark.parametrize('abits, tol', ([8, 1e-4], [16, 1e-2]))
    def test_mct_ptq_and_exporter_fq(self, abits, tol):
        quantized_model = self._run_mct(self.get_model(), self.representative_dataset(1), abits, mctq.QuantizationMethod.POWER_OF_TWO)
        onnx_model_dict = self._run_exporter(quantized_model, self.representative_dataset(1), QuantizationFormat.FAKELY_QUANT)

        self._assert_fq_quant_params_match(quantized_model, onnx_model_dict, mctq.QuantizationMethod.POWER_OF_TWO)
        self._assert_outputs_match(quantized_model, self.representative_dataset(1), QuantizationFormat.FAKELY_QUANT, tol=tol)

    @pytest.mark.parametrize('a_qmethod, tol', [(mctq.QuantizationMethod.POWER_OF_TWO, 0.0),
                                                (mctq.QuantizationMethod.SYMMETRIC, 1e-2),
                                                (mctq.QuantizationMethod.UNIFORM, 1e-2)])
    @pytest.mark.parametrize('abits', [8, 16])
    def test_mct_qat_and_exporter_mctq(self, abits, a_qmethod, tol):
        quantized_model = self._run_mct_qat(self.get_model(), self.representative_dataset(1), abits, a_qmethod)
        onnx_model_dict = self._run_exporter(quantized_model, self.representative_dataset(1), QuantizationFormat.MCTQ)

        self._assert_quant_params_match(quantized_model, onnx_model_dict, a_qmethod)
        self._assert_outputs_match(quantized_model, self.representative_dataset(1), QuantizationFormat.MCTQ, tol=tol)

    @pytest.mark.parametrize('abits, tol', ([8, 1e-8], [16, 1e-2]))
    def test_mct_qat_and_exporter_fq(self, abits, tol):
        quantized_model = self._run_mct_qat(self.get_model(), self.representative_dataset(1), abits, mctq.QuantizationMethod.POWER_OF_TWO)
        onnx_model_dict = self._run_exporter(quantized_model, self.representative_dataset(1), QuantizationFormat.FAKELY_QUANT)

        self._assert_fq_quant_params_match(quantized_model, onnx_model_dict, mctq.QuantizationMethod.POWER_OF_TWO)
        self._assert_outputs_match(quantized_model, self.representative_dataset(1), QuantizationFormat.FAKELY_QUANT, tol=tol)

    @pytest.mark.parametrize('abits', [8, 16])
    def test_multi_input_mct_and_exporter_mctq(self, abits):
        class MultiInputModel(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.linear = torch.nn.Linear(in_channels, out_channels)
                self.linear_y = torch.nn.Linear(in_channels, out_channels)

            def forward(self, x, y):
                return self.linear(x) + self.linear_y(y)

        quantized_model = self._run_mct(MultiInputModel(self.in_channels, self.out_channels), self.representative_dataset(2),
                                        abits, mctq.QuantizationMethod.POWER_OF_TWO)
        self._run_exporter(quantized_model, self.representative_dataset(2), QuantizationFormat.MCTQ)
        self._assert_outputs_match(quantized_model, self.representative_dataset(2), QuantizationFormat.MCTQ)

    @pytest.mark.parametrize('abits', [8, 16])
    def test_multi_output_mct_and_exporter_mctq(self, abits):
        class MultiOutputModel(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.linear = torch.nn.Linear(in_channels, out_channels)
                self.linear_y = torch.nn.Linear(in_channels, out_channels)

            def forward(self, x):
                return self.linear(x), self.linear_y(x)

        quantized_model = self._run_mct(MultiOutputModel(self.in_channels, self.out_channels), self.representative_dataset(1),
                                        abits, mctq.QuantizationMethod.POWER_OF_TWO)
        self._run_exporter(quantized_model, self.representative_dataset(1), QuantizationFormat.MCTQ)
        self._assert_outputs_match(quantized_model, self.representative_dataset(1), QuantizationFormat.MCTQ)

    @pytest.mark.parametrize('abits', [8, 16])
    def test_multi_input_output_mct_and_exporter_mctq(self, abits):
        class MultiInputOutputModel(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.linear = torch.nn.Linear(in_channels, out_channels)
                self.linear_y = torch.nn.Linear(in_channels, out_channels)

            def forward(self, x, y):
                return self.linear(x+y), self.linear_y(x*y)

        quantized_model = self._run_mct(MultiInputOutputModel(self.in_channels, self.out_channels), self.representative_dataset(2),
                                        abits, mctq.QuantizationMethod.POWER_OF_TWO)
        self._run_exporter(quantized_model, self.representative_dataset(2), QuantizationFormat.MCTQ)
        self._assert_outputs_match(quantized_model, self.representative_dataset(2), QuantizationFormat.MCTQ)

    @pytest.mark.parametrize('abits', ([2, 4]))
    def test_mct_ptq_and_exporter_mctq_lut(self, abits):
        quantized_model = self._run_mct(self.get_model(), self.representative_dataset(1), abits,
                                        mctq.QuantizationMethod.LUT_POT_QUANTIZER, w_qmethod=mctq.QuantizationMethod.LUT_SYM_QUANTIZER)
        onnx_model_dict = self._run_exporter(quantized_model, self.representative_dataset(1), QuantizationFormat.MCTQ)

        self._assert_outputs_match(quantized_model, self.representative_dataset(1), QuantizationFormat.MCTQ)
