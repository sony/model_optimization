import copy
from enum import Enum
from typing import List, Any
import numpy as np

from model_compression_toolkit import MixedPrecisionQuantizationConfig, DEFAULTCONFIG, get_model
from model_compression_toolkit.common.constants import TENSORFLOW
from model_compression_toolkit.hardware_models.default_hwm import generate_default_hardware_model
from model_compression_toolkit.hardware_models.keras_hardware_model.keras_default import generate_fhw_model_keras, \
    get_default_hwm_keras
from model_compression_toolkit.keras.constants import DEFAULT_HWM
from tests.common_tests.base_test import BaseTest


class LayerTestMode(Enum):
    FLOAT = 0
    QUANTIZED_8_BITS = 1


class BaseLayerTest(BaseTest):
    def __init__(self,
                 unit_test,
                 layers,
                 val_batch_size=1,
                 num_calibration_iter=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3),
                 quantization_modes: List[LayerTestMode] = [LayerTestMode.QUANTIZED_8_BITS, LayerTestMode.FLOAT],
                 is_inputs_a_list=False,
                 use_cpu=False):

        super().__init__(unit_test,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape)

        self.use_cpu = use_cpu
        self.is_inputs_a_list = is_inputs_a_list
        assert isinstance(quantization_modes, list) and len(quantization_modes) > 0
        self.quantization_modes = quantization_modes
        self.current_mode = None
        assert isinstance(layers, list) and len(layers) > 0, f'Layers list can not be empty'
        self.layers = layers

    def predict(self, model: Any, input: List[np.ndarray]):
        raise Exception(f'Predict should be implemented in framework layer test.')

    def get_layers(self):
        return self.layers

    def get_fw_hw_model(self):
        if self.current_mode == LayerTestMode.FLOAT:
            # Disable all features that are enabled by default:
            hwm = generate_default_hardware_model(enable_weights_quantization=False,
                                                  enable_activation_quantization=False)
            return generate_fhw_model_keras(name="base_layer_test", hardware_model=hwm)
        elif self.current_mode == LayerTestMode.QUANTIZED_8_BITS:
            return get_default_hwm_keras()
        else:
            raise NotImplemented

    def get_quantization_config(self):
        qc = copy.deepcopy(DEFAULTCONFIG)
        qc.weights_bias_correction = False
        return qc

    def run_test(self):
        feature_networks = self.create_networks()
        feature_networks = feature_networks if isinstance(feature_networks, list) else [feature_networks]
        for model_float in feature_networks:
            for mode in self.quantization_modes:
                self.current_mode = mode
                qc = self.get_quantization_config()
                if isinstance(qc, MixedPrecisionQuantizationConfig):
                    ptq_model, quantization_info = self.get_mixed_precision_ptq_facade()(model_float,
                                                                                         self.representative_data_gen,
                                                                                         n_iter=self.num_calibration_iter,
                                                                                         quant_config=qc,
                                                                                         fw_info=self.get_fw_info(),
                                                                                         fw_hw_model=self.get_fw_hw_model())
                else:
                    ptq_model, quantization_info = self.get_ptq_facade()(model_float,
                                                                         self.representative_data_gen,
                                                                         n_iter=self.num_calibration_iter,
                                                                         quant_config=qc,
                                                                         fw_info=self.get_fw_info(),
                                                                         fw_hw_model=self.get_fw_hw_model())

                self.compare(ptq_model, model_float, input_x=self.representative_data_gen(),
                             quantization_info=quantization_info)

