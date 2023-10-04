from typing import Dict, List

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import HessianRequest
from model_compression_toolkit.core.common.hessian.hessian_calculator import HessianCalculator
import tensorflow as tf

from model_compression_toolkit.core.common.hessian.hessian_config import HessianConfig

class HessianCalculatorKeras(HessianCalculator):
    """
    Hessian calculator for Keras graphs.
    """

    def __init__(self,
                 graph: Graph,
                 hessian_config: HessianConfig,
                 input_images: List[tf.Tensor],
                 fw_impl,
                 hessian_request: HessianRequest):

        super(HessianCalculatorKeras, self).__init__(graph=graph,
                                                     hessian_config=hessian_config,
                                                     input_images=input_images,
                                                     fw_impl=fw_impl,
                                                     hessian_request=hessian_request)


