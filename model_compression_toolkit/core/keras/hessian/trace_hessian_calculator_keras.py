from typing import Dict, List

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian import TraceHessianRequest
from model_compression_toolkit.core.common.hessian.trace_hessian_calculator import TraceHessianCalculator
import tensorflow as tf

from model_compression_toolkit.core.common.hessian.trace_hessian_config import TraceHessianConfig

class TraceHessianCalculatorKeras(TraceHessianCalculator):
    """
    Hessian calculator for Keras graphs.
    """

    def __init__(self,
                 graph: Graph,
                 trace_hessian_config: TraceHessianConfig,
                 input_images: List[tf.Tensor],
                 fw_impl,
                 trace_hessian_request: TraceHessianRequest):

        super(TraceHessianCalculatorKeras, self).__init__(graph=graph,
                                                          trace_hessian_config=trace_hessian_config,
                                                          input_images=input_images,
                                                          fw_impl=fw_impl,
                                                          trace_hessian_request=trace_hessian_request)


