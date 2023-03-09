import copy
from enum import Enum
from typing import List, Any
import numpy as np

from model_compression_toolkit import QuantizationConfig, QuantizationErrorMethod, CoreConfig
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

    def get_tpc(self):
        raise NotImplemented

    def get_quantization_config(self):
        return QuantizationConfig(weights_bias_correction=False)

    def run_test(self):
        feature_networks = self.create_networks()
        feature_networks = feature_networks if isinstance(feature_networks, list) else [feature_networks]
        for model_float in feature_networks:
            for mode in self.quantization_modes:
                self.current_mode = mode
                core_config = self.get_core_config()
                ptq_model, quantization_info = self.get_ptq_facade()(model_float,
                                                                     self.representative_data_gen_experimental,
                                                                     core_config=core_config,
                                                                     target_platform_capabilities=self.get_tpc(),
                                                                     new_experimental_exporter=True)

                self.compare(ptq_model, model_float, input_x=self.representative_data_gen(),
                             quantization_info=quantization_info)

