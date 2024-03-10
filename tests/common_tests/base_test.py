from typing import Any, List

from model_compression_toolkit.core import CoreConfig, DebugConfig, QuantizationConfig
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
import numpy as np

from model_compression_toolkit.core.common.user_info import UserInformation


class BaseTest:
    def __init__(self, unit_test,
                 val_batch_size=1,
                 num_calibration_iter=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3)):

        self.unit_test = unit_test
        self.val_batch_size = val_batch_size
        self.num_calibration_iter = num_calibration_iter
        self.num_of_inputs = num_of_inputs
        self.input_shape = (val_batch_size,) + input_shape

    def generate_inputs(self):
        return [np.random.randn(*in_shape) for in_shape in self.get_input_shapes()]

    def representative_data_gen(self):
        return self.generate_inputs()

    def representative_data_gen_experimental(self):
        for _ in range(self.num_calibration_iter):
            yield self.generate_inputs()

    def get_input_shapes(self):
        return [self.input_shape for _ in range(self.num_of_inputs)]

    def get_core_config(self):
        return CoreConfig(quantization_config=self.get_quantization_config(),
                          mixed_precision_config=self.get_mixed_precision_v2_config(),
                          debug_config=self.get_debug_config())

    def get_quantization_config(self):
        return QuantizationConfig()

    def get_mixed_precision_v2_config(self):
        return None

    def get_debug_config(self):
        return DebugConfig()

    def get_fw_impl(self) -> FrameworkImplementation:
        raise Exception('get_fw_impl is not implemented')

    def get_fw_info(self):
        raise Exception('get_fw_info is not implemented')

    def get_ptq_facade(self):
        raise NotImplementedError(f'{self.__class__} did not implement get_ptq_facade')

    def get_gptq_facade(self):
        raise NotImplementedError(f'{self.__class__} did not implement get_gptq_facade')

    def create_networks(self):
        raise Exception('create_networks is not implemented')

    def get_tpc(self):
        raise Exception('get_tpc is not implemented')

    def compare(self, ptq_model: Any,
                model_float: Any,
                input_x: List[np.ndarray],
                quantization_info: UserInformation):
        raise Exception('compare is not implemented')

    def run_test(self):
        raise Exception('run_test is not implemented')
