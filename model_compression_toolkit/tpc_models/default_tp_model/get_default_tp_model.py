from model_compression_toolkit.common.target_platform import TargetPlatformModel
import pkgutil
import importlib


def get_default_tp_model(version: str) -> TargetPlatformModel:
    """
    A method that generates a default target platform model, with base 8-bit quantization configuration and 8, 4, 2
    bits configuration list for mixed-precision quantization.
    NOTE: in order to generate a target platform model with different configurations but with the same Operators Sets
    (for tests, experiments, etc.), use this method implementation as a test-case, i.e., override the
    'get_op_quantization_configs' method and use its output to call 'generate_tp_model' with your configurations.

    Returns: A TargetPlatformModel object.

    """
    current_pkg_path = '/'.join(__file__.split('/')[:-1])
    if version is None: # Get latest
        version_to_generate = max([name for _, name, _ in pkgutil.iter_modules([current_pkg_path])])
    else:
        # TODO: specific version handle

    m = importlib.import_module('model_compression_toolkit.tpc_models.default_tp_model.v2_3_0')
    base_config, mixed_precision_cfg_list = m.get_op_quantization_configs()
    return m.generate_tp_model(default_config=base_config,
                             base_config=base_config,
                             mixed_precision_cfg_list=mixed_precision_cfg_list,
                             name='default_tp_model')