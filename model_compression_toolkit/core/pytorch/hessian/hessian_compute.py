from model_compression_toolkit.core.common.hessian.hessian_compute import HessianCalculator


class HessianCalculatorPytorch(HessianCalculator):

    def __init__(self, model, graph, config, input_data):
        super(HessianCalculatorPytorch, self).__init__(model=model,
                                                       graph=graph,
                                                       config=config,
                                                       input_data=input_data)

    def compute(self):
        pass
