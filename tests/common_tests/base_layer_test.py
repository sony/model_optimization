import copy
from enum import Enum
from typing import List

from model_compression_toolkit import MixedPrecisionQuantizationConfig, DEFAULTCONFIG
from tests.common_tests.base_test import BaseTest


class LayerTestMode(Enum):
    FLOAT = 0
    QUANTIZED_8_BITS = 1


class BaseLayerTest(BaseTest):
    def __init__(self,
                 unit_test,
                 val_batch_size=1,
                 num_calibration_iter=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3),
                 quantization_modes: List[LayerTestMode] = [LayerTestMode.QUANTIZED_8_BITS, LayerTestMode.FLOAT],
                 is_inputs_a_list=False):

        super().__init__(unit_test,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape)

        self.is_inputs_a_list = is_inputs_a_list
        assert isinstance(quantization_modes, list) and len(quantization_modes) > 0
        self.quantization_modes = quantization_modes
        self.current_mode = None

    def get_layers(self):
        raise Exception('Implement it')

    def get_quantization_config(self):
        qc = copy.deepcopy(DEFAULTCONFIG)
        # Disable all features that are enabled by default:
        qc.weights_bias_correction = False
        if self.current_mode == LayerTestMode.FLOAT:
            qc.enable_activation_quantization = False
            qc.enable_weights_quantization = False
        elif self.current_mode == LayerTestMode.QUANTIZED_8_BITS:
            qc.weights_n_bits = 8
            qc.activation_n_bits = 8
        else:
            raise NotImplemented
        return qc

    def run_test(self):
        feature_networks = self.create_networks()
        feature_networks = feature_networks if isinstance(feature_networks, list) else [feature_networks]
        for model_float in feature_networks:
            for mode in self.quantization_modes:
                self.current_mode = mode
                print(f'Mode: {self.current_mode}')
                qc = self.get_quantization_config()
                if isinstance(qc, MixedPrecisionQuantizationConfig):
                    ptq_model, quantization_info = self.get_mixed_precision_ptq_facade()(model_float,
                                                                                         self.representative_data_gen,
                                                                                         n_iter=self.num_calibration_iter,
                                                                                         quant_config=qc,
                                                                                         fw_info=self.get_fw_info())
                else:
                    ptq_model, quantization_info = self.get_ptq_facade()(model_float,
                                                                         self.representative_data_gen,
                                                                         n_iter=self.num_calibration_iter,
                                                                         quant_config=qc,
                                                                         fw_info=self.get_fw_info())

                self.compare(ptq_model, model_float, input_x=self.representative_data_gen(),
                             quantization_info=quantization_info)
