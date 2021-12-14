import copy
from enum import Enum
from typing import Any, List

from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    DEFAULT_MIXEDPRECISION_CONFIG, MixedPrecisionQuantizationConfig
import numpy as np

from model_compression_toolkit.common.user_info import UserInformation


class TestMode(Enum):
    FLOAT = 0
    QUANTIZED_16_BITS = 1


class BaseTest:
    def __init__(self, unit_test,
                 val_batch_size=1,
                 num_calibration_iter=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3),
                 quantization_modes: List[TestMode] = None):

        self.unit_test = unit_test
        self.val_batch_size = val_batch_size
        self.num_calibration_iter = num_calibration_iter
        self.num_of_inputs = num_of_inputs
        self.input_shape = (val_batch_size,) + input_shape
        assert isinstance(quantization_modes, list) and len(quantization_modes) > 0
        self.quantization_modes = quantization_modes
        self.quantization_mode = None

    def generate_inputs(self):
        return [np.random.randn(*in_shape) for in_shape in self.get_input_shapes()]

    def get_input_shapes(self):
        return [self.input_shape for _ in range(self.num_of_inputs)]

    def get_quantization_config(self):
        qc = copy.deepcopy(DEFAULTCONFIG)
        # Disable all features that are enabled by default:
        qc.weights_bias_correction = False
        if self.quantization_mode == TestMode.FLOAT:
            qc.enable_activation_quantization = False
            qc.enable_weights_quantization = False
        elif self.quantization_mode == TestMode.QUANTIZED_16_BITS:
            qc.weights_n_bits = 16
            qc.activation_n_bits = 16
        else:
            raise NotImplemented
        return qc

    def get_mixed_precision_quantization_config(self):
        return DEFAULT_MIXEDPRECISION_CONFIG

    def get_fw_impl(self) -> FrameworkImplementation:
        raise Exception('get_fw_impl is not implemented')

    def get_fw_info(self):
        raise Exception('get_fw_info is not implemented')

    def get_ptq_facade(self):
        raise NotImplementedError(f'{self.__class__} did not implement get_ptq_facade')

    def get_mixed_precision_ptq_facade(self):
        raise NotImplementedError(f'{self.__class__} did not implement get_mixed_precision_ptq_facade')

    def create_networks(self):
        raise Exception('create_networks is not implemented')

    def run_test(self):
        raise Exception('run_test is not implemented')

    def compare(self, ptq_model: Any,
                model_float: Any,
                input_x: List[np.ndarray],
                quantization_info: UserInformation):
        raise Exception('compare is not implemented')
