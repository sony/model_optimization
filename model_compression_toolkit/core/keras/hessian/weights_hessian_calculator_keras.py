from typing import List

import tensorflow as tf

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian.hessian_config import HessianConfig
from model_compression_toolkit.core.keras.hessian.hessian_calculator_keras import HessianCalculatorKeras


class WeightsHessianCalculatorKeras(HessianCalculatorKeras):

    def __init__(self,
                 graph: Graph,
                 config: HessianConfig,
                 input_images: tf.Tensor):
        super(WeightsHessianCalculatorKeras, self).__init__(graph=graph,
                                                            config=config,
                                                            input_images=input_images)

    def compute(self):
        raise NotImplemented  # TODO: implement computation for all granularities
