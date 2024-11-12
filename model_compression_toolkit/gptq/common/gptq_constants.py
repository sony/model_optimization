# Parameters names
AUXVAR = 'auxvar_tensor'
ITERVAR = 'iteration_variable'
SCALE_TENSOR = "scale_ptq_tensor"
AUXSHIFT = 'shift'
WEIGHTS_QUANTIZATION_PARAMS = 'weights_quantization_params'
PTQ_MIN_RANGE = "min_range"
PTQ_MAX_RANGE = "max_range"
PTQ_THRESHOLD = "ptq_threshold"
SCALE_PTQ = "scale"

# Default quantizer values
N_CYCLES = 4
MIM_TEMP = 0.5
MAX_TEMP = 1.0
REG_DEFAULT = 0.01
REG_DEFAULT_SLA = 10
MAX_LSB_CHANGE = 1

# Soft rounding arguments values
SOFT_ROUNDING_GAMMA = -0.1
SOFT_ROUNDING_ZETA = 1.1

# GPTQ config constant
QUANT_PARAM_LEARNING_STR = 'quantization_parameter_learning'
MAX_LSB_STR = 'max_lsbs_change_map'

# GPTQ learning hyperparameters
LR_DEFAULT = 3e-2
LR_REST_DEFAULT = 1e-4
LR_BIAS_DEFAULT = 1e-4
GPTQ_MOMENTUM = 0.9
