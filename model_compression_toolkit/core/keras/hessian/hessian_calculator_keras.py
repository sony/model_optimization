from typing import Dict, List

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian.hessian_calculator import HessianCalculator
import tensorflow as tf

from model_compression_toolkit.core.common.hessian.hessian_config import HessianConfig

class HessianCalculatorKeras(HessianCalculator):

    def __init__(self,
                 graph: Graph,
                 config: HessianConfig,
                 input_images: List[tf.Tensor],
                 fw_impl):

        super(HessianCalculatorKeras, self).__init__(graph=graph,
                                                     config=config,
                                                     input_images=input_images,
                                                     fw_impl=fw_impl)


    # def compute(self):
    #     calculator=None
    #     if self.config.mode == HessianMode.ACTIVATIONS:
    #         calculator = ActivationHessianCalculatorKeras(graph=self.graph,
    #                                                       config=self.config,
    #                                                       input_data=self.input_data)
    #     elif self.config.mode == HessianMode.WEIGHTS:
    #         calculator = WeightsHessianCalculatorKeras(graph=self.graph,
    #                                                    config=self.config,
    #                                                    input_data=self.input_data)
    #     else:
    #         Logger.error(f"Not supported mode {self.config.mode}")
    #     return calculator.compute()


