from typing import Tuple
import importlib


def find_modules(lib: str) -> Tuple[str, str]:
    """
    Finds the relevant module paths for a given pre-trained models library.

    Args:
        lib (str): Name of the library.

    Returns:
        Tuple of model_lib_module and quant_module, representing the module paths needed to be imported
        for the given library.

    Raises:
        Error: If the specified library is not found.
    """

    # Search in PyTorch libraries
    if importlib.util.find_spec('pytorch_fw.' + lib) is not None:
        model_lib_module = 'pytorch_fw.' + lib + '.model_lib_' + lib
        quant_module = 'pytorch_fw.quant'

    # Search in Keras libraries
    elif importlib.util.find_spec('keras_fw.' + lib) is not None:
        model_lib_module = 'keras_fw.' + lib + '.model_lib_' + lib
        quant_module = 'keras_fw.quant'
    else:
        raise Exception(f'Error: model library {lib} is not supported')
    return model_lib_module, quant_module
