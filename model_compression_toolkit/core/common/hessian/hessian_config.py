from enum import Enum


class HessianMode(Enum):
    WEIGHTS = 0
    ACTIVATIONS = 1


class HessianGranularity(Enum):
    PER_ELEMENT = 0
    PER_OUTPUT_CHANNEL = 1
    PER_LAYER = 2


class HessianConfig:
    def __init__(self,
                 mode: HessianMode,
                 granularity: HessianGranularity,
                 nodes_names_for_hessian_computation, # interest points
                 alpha: float = 0.3,
                 num_iterations: int = 50,
                 norm_weights: bool = True,
                 search_output_replacement: bool = False
                 ):

        self.mode = mode  # activations or weights
        self.granularity = granularity  # per element, per layer, per channel
        self.nodes_names_for_hessian_computation = nodes_names_for_hessian_computation
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.norm_weights = norm_weights
        self.search_output_replacement = search_output_replacement

