## Introduction

Export your quantized model in the following serialization formats:

* TensorFlow models can be exported as Tensorflow models (`.h5` or `.keras` extensions) and TFLite models (`.tflite` extension).
* PyTorch models can be exported as torch script models and ONNX models (`.onnx` extension).

You can export your quantized model in the following quantization formats:
* Fake Quant (where weights and activations are float fakely-quantized values)
* INT8 (where weights and activations are represented using 8bits integers)
* MCTQ (where weights and activations are quantized using mct-quantizers custom quantizers).


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


### keras/h5

The model will be exported as a tensorflow `.keras` or `.h5` model (depends the utilized Tensorflow version)
where weights and activations are quantized but represented using a float32 dtype (fakely-quant).

#### Usage Example

#### MCTQ

By default, `mct.exporter.keras_export_model` will export the quantized Keras model to a .keras/.h5 model with custom quantizers from mct_quantizers module.

```python
import tempfile

# Path of exported model
_, keras_file_path = tempfile.mkstemp('.keras')

# Export a keras model with mctq custom quantizers.
mct.exporter.keras_export_model(model=quantized_exportable_model, 
                                save_model_path=keras_file_path)
```

Notice that the fakely-quantized model has the same size as the quantized exportable model as weights data types are
float.

#### Fakely-Quantized

```python
# Path of exported model
_, keras_file_path = tempfile.mkstemp('.keras')

# Use mode KerasExportSerializationFormat.KERAS_H5 for a .keras/.h5 model 
# and QuantizationFormat.FAKELY_QUANT for fakely-quantized weights 
# and activations.
mct.exporter.keras_export_model(model=quantized_exportable_model, 
                                save_model_path=keras_file_path,
                                quantization_format=mct.exporter.QuantizationFormat.FAKELY_QUANT)
```

Notice that the fakely-quantized model has the same size as the quantized exportable model as weights data types are
float.


### TFLite
The tflite serialization format export in two qauntization formats 

#### INT8 TFLite

The model will be exported as a tflite model where weights and activations are represented as 8bit integers.

##### Usage Example

```python

import tempfile

# Path of exported model
_, tflite_file_path = tempfile.mkstemp('.tflite')

# Use mode KerasExportSerializationFormat.TFLITE for tflite model and quantization_format.INT8.
mct.exporter.keras_export_model(model=quantized_exportable_model,
                                save_model_path=tflite_file_path,
                                serialization_format=mct.exporter.KerasExportSerializationFormat.TFLITE,
                                quantization_format=mct.exporter.QuantizationFormat.INT8)

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

#### Fakely-Quantized TFLite

The model will be exported as a tflite model where weights and activations are quantized but represented as float.
operators.

##### Usage Example

```python

# Path of exported model
_, tflite_file_path = tempfile.mkstemp('.tflite')

# Use mode KerasExportSerializationFormat.TFLITE for tflite model and QuantizationFormat.FAKELY_QUANT for fakely-quantized weights 
# and activations.
mct.exporter.keras_export_model(model=quantized_exportable_model,
                                save_model_path=tflite_file_path,
                                serialization_format=mct.exporter.KerasExportSerializationFormat.TFLITE,
                                quantization_format=mct.exporter.QuantizationFormat.FAKELY_QUANT)
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

### ONNX

The model will be exported in ONNX format where weights and activations are quantized 
using mct_quantizers custom quantization ops, or represented as float 
(fakely quant).

#### Usage Example


#### MCTQ

By default, `mct.exporter.pytorch_export_model` will export the quantized pytorch model to an onnx model with custom quantizers from mct_quantizers module.  

```python
import tempfile

# Path of exported model
_, onnx_file_path = tempfile.mkstemp('.onnx')

# Export onnx model with mctq quantizers.
mct.exporter.pytorch_export_model(model=quantized_exportable_model,
                                  save_model_path=onnx_file_path,
                                  repr_dataset=representative_data_gen)
```


#### Fakely-Quant

For exporting a fakely-quantized model, use QuantizationFormat.FAKELY_QUANT:
```python
import tempfile

# Path of exported model
_, onnx_file_path = tempfile.mkstemp('.onnx')

# Use QuantizationFormat.FAKELY_QUANT for fakely-quantized weights and activations.
mct.exporter.pytorch_export_model(model=quantized_exportable_model,
                                  save_model_path=onnx_file_path,
                                  repr_dataset=representative_data_gen,
                                  quantization_format=mct.exporter.QuantizationFormat.FAKELY_QUANT)
```

Notice that the fakely-quantized model has the same size as the quantized 
exportable model as weights data types are float.

### TorchScript

The model will be exported in TorchScript format where weights and activations are quantized but represented as float 
(fakely quant).

#### Usage Example

```python
# Path of exported model
_, torchscript_file_path = tempfile.mkstemp('.pt')


# Use mode PytorchExportSerializationFormat.TORCHSCRIPT a torchscript model 
# and QuantizationFormat.FAKELY_QUANT for fakely-quantized weights and activations.
mct.exporter.pytorch_export_model(model=quantized_exportable_model,
                                  save_model_path=torchscript_file_path,
                                  repr_dataset=representative_data_gen,
                                  serialization_format=mct.exporter.PytorchExportSerializationFormat.TORCHSCRIPT,
                                  quantization_format=mct.exporter.QuantizationFormat.FAKELY_QUANT)
```

Notice that the fakely-quantized model has the same size as the quantized exportable model as weights data types are
float.
