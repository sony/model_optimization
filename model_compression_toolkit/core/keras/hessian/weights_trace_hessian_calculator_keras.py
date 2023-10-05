from typing import List, Dict

import tensorflow as tf

from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.hessian.trace_hessian_config import TraceHessianConfig
from model_compression_toolkit.core.keras.hessian.trace_hessian_calculator_keras import TraceHessianCalculatorKeras


class WeightsTraceHessianCalculatorKeras(TraceHessianCalculatorKeras):
    """
    Hessian w.r.t weights for Keras graph computation.
    """

    def __init__(self,
                 graph: Graph,
                 config: TraceHessianConfig,
                 input_images: List[tf.Tensor],
                 fw_impl):
        """

        Args:
            graph: Float graph to compute its hessian data.
            config: HessianConfig to use for during Hessian computation.
            input_images: List of images to use the the computaion (image per graph input).
            fw_impl: Framework implementation to use during computation.
        """

        super(WeightsTraceHessianCalculatorKeras, self).__init__(graph=graph,
                                                                 config=config,
                                                                 input_images=input_images,
                                                                 fw_impl=fw_impl)

    def compute(self) -> Dict[BaseNode, float]:
        """
        Compute the hessian of the float graph based on the configuration and images that in
        the calculator.

        Returns: Dictionary from interest point to hessian score.

        """
        raise NotImplemented