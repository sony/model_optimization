# Parameters names
AUXVAR = 'auxvar_tensor'
ITERVAR = 'iteration_variable'
THRESHOLD_TENSOR = "ptq_threshold_tensor"
SCALE_TENSOR = "scale_ptq_tensor"
GPTQ_ITER = "_gptq_iter"
AUXSHIFT = '_shift'
TEMP = '_temp'
WEIGHTS_QUANTIZATION_PARAMS = 'weights_quantization_params'
PTQ_MIN_RANGE = "_min_range"
PTQ_MAX_RANGE = "_max_range"
PTQ_THRESHOLD = "_ptq_threshold"
SCALE_PTQ = "_scale"

# Default quantizer values
N_EPOCHS = 10000
N_CYCLES = 4
MIM_TEMP = 0.5
MAX_TEMP = 1.0
REG_DEFAULT = 0.01
MAX_ITERATIONS_DEFAULT = 10000

# Soft rounding arguments values
SOFT_ROUNDING_GAMMA = -0.1
SOFT_ROUNDING_ZETA = 1.1
SOFT_ROUNDING_BETA = 2 / 3

# GPTQ config constant
REGULARIZATION_FUNCTION = "regularization_function"
