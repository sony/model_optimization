## Introduction

Export your quantized model in the following formats:

* TensorFlow models can be exported as Tensorflow models (`.h5` extension) and TFLite models (`.tflite` extension).
* PyTorch models can be exported as torch script models and ONNX models (`.onnx` extension).

### Note

This feature is **experimental and subject to future changes**. If you have any questions or issues,
please [open an issue](https://github.com/sony/model_optimization/issues/new/choose) in this GitHub repository.

## Export TensorFlow Models

To export a TensorFlow model - a quantized model from MCT should be quantized first:

```python
import numpy as np
from keras.applications import ResNet50
import model_compression_toolkit as mct

# Create a model
float_model = ResNet50()
# Quantize the model. In order to export the model set new_experimental_exporter to True.
# Notice that here the representative dataset is random. 
quantized_exportable_model, _ = mct.ptq.keras_post_training_quantization_experimental(float_model,
                                                                                  representative_data_gen=lambda: [
                                                                                      np.random.random(
                                                                                          (1, 224, 224, 3))],
                                                                                  new_experimental_exporter=True)
```

### INT8 TFLite

The model will be exported as a tflite model where weights and activations are represented as 8bit integers.

#### Usage Example

```python
import tempfile

from mct.target_platform_capabilities.tpc_models.tflite_tpc.latest import get_keras_tpc_latest
from mct.exporter.model_exporter.fw_agonstic.export_serialization_format import ExportSerializationFormat

# Path of exported model
_, tflite_file_path = tempfile.mkstemp('.tflite')

# Get TPC
tflite_int8_tpc = get_keras_tpc_latest()

# Use mode ExportSerializationFormat.TFLITE for tflite model and tflite keras tpc for INT8 data type.
mct.exporter.keras_export_model(model=quantized_exportable_model, save_model_path=tflite_file_path,
                                target_platform_capabilities=tflite_int8_tpc,
                                serialization_format=ExportSerializationFormat.TFLITE)
```

Compare size of float and quantized model:

```python
import os

# Save float model to measure its size
_, float_file_path = tempfile.mkstemp('.h5')
float_model.save(float_file_path)

print("Float model in Mb:", os.path.getsize(float_file_path) / float(2 ** 20))
print("Quantized model in Mb:", os.path.getsize(tflite_file_path) / float(2 ** 20))
print(f'Compression ratio: {os.path.getsize(float_file_path) / os.path.getsize(tflite_file_path)}')
```

### Fakely-Quantized TFLite

The model will be exported as a tflite model where weights and activations are quantized but represented as float.
Notice that activation quantizers are implemented
using [tf.quantization.fake_quant_with_min_max_vars](https://www.tensorflow.org/api_docs/python/tf/quantization/fake_quant_with_min_max_vars)
operators.

#### Usage Example

```python

from mct.target_platform_capabilities.tpc_models.default_tpc.latest import get_keras_tpc_latest
from mct.exporter.model_exporter.fw_agonstic.export_serialization_format import ExportSerializationFormat

# Path of exported model
_, tflite_file_path = tempfile.mkstemp('.tflite')

# Get TPC
keras_tpc = get_keras_tpc_latest()

# Use mode ExportSerializationFormat.TFLITE for tflite model and keras tpc for fakely-quantized weights 
# and activations
mct.exporter.tflite_export_model(model=quantized_exportable_model,
                                 save_model_path=tflite_file_path,
                                 target_platform_capabilities=keras_tpc,
                                 serialization_format=ExportSerializationFormat.TFLITE)
```

Notice that the fakely-quantized model has the same size as the quantized exportable model as weights data types are
float.


### Fakely-Quantized h5

The model will be exported as a tensorflow `.h5` model where weights and activations are quantized but represented as
float.

#### Usage Example

```python
import tempfile

from mct.target_platform_capabilities.tpc_models.default_tpc.latest import get_keras_tpc_latest
from mct.exporter.model_exporter.fw_agonstic.export_serialization_format import ExportSerializationFormat

# Path of exported model
_, h5_file_path = tempfile.mkstemp('.h5')

# Get TPC
keras_tpc = get_keras_tpc_latest()

# Use mode ExportSerializationFormat.KERAS_H5 for keras h5 model and default keras tpc for fakely-quantized weights 
# and activations
mct.exporter.keras_export_model(model=quantized_exportable_model, save_model_path=h5_file_path,
                                target_platform_capabilities=keras_tpc,
                                serialization_format=ExportSerializationFormat.KERAS_H5)
```

Notice that the fakely-quantized model has the same size as the quantized exportable model as weights data types are
float.

## Export PyTorch models

To export a PyTorch model - a quantized model from MCT should be quantized first:

```python
import model_compression_toolkit as mct
import numpy as np
import torch
from torchvision.models.mobilenetv2 import mobilenet_v2

# Create a model
float_model = mobilenet_v2()


# Quantize the model. In order to export the model set new_experimental_exporter to True.
# Notice that here the representative dataset is random.
def representative_data_gen():
    for i in range(1):
        yield [np.random.random((1, 3, 224, 224))]


quantized_exportable_model, _ = mct.ptq.pytorch_post_training_quantization_experimental(float_model,
                                                                                    representative_data_gen=representative_data_gen,
                                                                                    new_experimental_exporter=True)
```

### Fakely-Quantized ONNX

The model will be exported in ONNX format where weights and activations are quantized but represented as float.

#### Usage Example

```python
import tempfile

from mct.target_platform_capabilities.tpc_models.default_tpc.latest import get_pytorch_tpc_latest
from mct.exporter.model_exporter.fw_agonstic.export_serialization_format import ExportSerializationFormat

# Path of exported model
_, onnx_file_path = tempfile.mkstemp('.onnx')

# Get TPC
pytorch_tpc = get_pytorch_tpc_latest()

# Use mode ExportSerializationFormat.ONNX for keras h5 model and default pytorch tpc for fakely-quantized weights 
# and activations
mct.exporter.pytorch_export_model(model=quantized_exportable_model, save_model_path=onnx_file_path,
                                  repr_dataset=representative_data_gen, target_platform_capabilities=pytorch_tpc,
                                  serialization_format=ExportSerializationFormat.ONNX)
```

Notice that the fakely-quantized model has the same size as the quantized exportable model as weights data types are
float.

### Fakely-Quantized TorchScript

The model will be exported in TorchScript format where weights and activations are quantized but represented as float.

#### Usage Example

```python
import tempfile

from mct.target_platform_capabilities.tpc_models.default_tpc.latest import get_pytorch_tpc_latest
from mct.exporter.model_exporter.fw_agonstic.export_serialization_format import ExportSerializationFormat

# Path of exported model
_, torchscript_file_path = tempfile.mkstemp('.pt')

# Get TPC
pytorch_tpc = get_pytorch_tpc_latest()

# Use mode ExportSerializationFormat.TORCHSCRIPT for keras h5 model and default pytorch tpc for fakely-quantized weights 
# and activations
mct.exporter.pytorch_export_model(model=quantized_exportable_model, save_model_path=torchscript_file_path,
                                  repr_dataset=representative_data_gen, target_platform_capabilities=pytorch_tpc,
                                  serialization_format=ExportSerializationFormat.TORCHSCRIPT)
```

Notice that the fakely-quantized model has the same size as the quantized exportable model as weights data types are
float.
