import importlib
from packaging import version

from model_compression_toolkit.constants import TENSORFLOW

FOUND_TF = importlib.util.find_spec(TENSORFLOW) is not None

if FOUND_TF:
    import tensorflow as tf
    # MCT doesn't support TensorFlow version 2.16 or higher
    if version.parse(tf.__version__) >= version.parse("2.16"):
        FOUND_TF = False

FOUND_TORCH = importlib.util.find_spec("torch") is not None
FOUND_TORCHVISION = importlib.util.find_spec("torchvision") is not None
FOUND_ONNX = importlib.util.find_spec("onnx") is not None
FOUND_ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None
FOUND_SONY_CUSTOM_LAYERS = importlib.util.find_spec('sony_custom_layers') is not None
