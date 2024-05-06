:orphan:

.. _ug-exporter:


=================================
exporter Module
=================================

Allows to export a quantized model in different serialization formats and quantization formats.
For more details about the export formats and options, please refer to the project's GitHub `README file <https://github.com/sony/model_optimization/tree/main/model_compression_toolkit/exporter>`_.
If you have any questions or issues, please open an issue in this GitHub repository.


QuantizationFormat
==========================

.. autoclass:: model_compression_toolkit.exporter.QuantizationFormat


KerasExportSerializationFormat
================================
Select the serialization format for exporting a quantized Keras model.

.. autoclass:: model_compression_toolkit.exporter.KerasExportSerializationFormat


keras_export_model
==========================
Allows to export a Keras model that was quantized via MCT.

.. autoclass:: model_compression_toolkit.exporter.keras_export_model


Keras Tutorial
==========================
To export a TensorFlow model as a quantized model, it is necessary to first apply quantization
to the model using MCT:

.. code-block:: python

    import numpy as np
    from keras.applications import ResNet50
    import model_compression_toolkit as mct

    # Create a model
    float_model = ResNet50()
    # Quantize the model.
    # Notice that here the representative dataset is random for demonstration only.
    quantized_exportable_model, _ = mct.ptq.keras_post_training_quantization(float_model,
                                                                             representative_data_gen=lambda: [np.random.random((1, 224, 224, 3))])



++++++++++++++++++++++++++++
keras serialization format
++++++++++++++++++++++++++++
The model will be exported as a tensorflow `.keras` model where weights and activations are quantized but represented using a float32 dtype.
Two optional quantization formats are available: MCTQ and FAKELY_QUANT.

++++
MCTQ
++++

By default, `mct.exporter.keras_export_model` will export the quantized Keras model to
a .keras model with custom quantizers from mct_quantizers module.

.. code-block:: python

    import tempfile

    # Path of exported model
    _, keras_file_path = tempfile.mkstemp('.keras')

    # Export a keras model with mctq custom quantizers.
    mct.exporter.keras_export_model(model=quantized_exportable_model,
                                    save_model_path=keras_file_path)

Notice that the model has the same size as the quantized exportable model as weights data types are float.



PytorchExportSerializationFormat
==================================
Select the serialization format for exporting a quantized Pytorch model.

.. autoclass:: model_compression_toolkit.exporter.PytorchExportSerializationFormat





pytorch_export_model
==========================
Allows to export a Pytorch model that was quantized via MCT.

.. autoclass:: model_compression_toolkit.exporter.pytorch_export_model



Pytorch Tutorial
==========================

In order to export your quantized model to ONNX format, and use it for inference, some additional packages are needed. Notice, this is needed only for models exported to ONNX format, so this part can be skipped if this is not planned:

.. code-block:: shell

    ! pip install -q onnx onnxruntime onnxruntime-extensions

Now, let's start the export demonstration by quantizing the model using MCT:

.. code-block:: python

    import model_compression_toolkit as mct
    import numpy as np
    import torch
    from torchvision.models.mobilenetv2 import mobilenet_v2

    # Create a model
    float_model = mobilenet_v2()


    # Notice that here the representative dataset is random for demonstration only.
    def representative_data_gen():
        yield [np.random.random((1, 3, 224, 224))]


    quantized_exportable_model, _ = mct.ptq.pytorch_post_training_quantization(float_model, representative_data_gen=representative_data_gen)



+++++++++
ONNX
+++++++++

The model will be exported in ONNX format where weights and activations are represented as float. Notice that `onnx` should be installed in order to export the model to an ONNX model.

There are two optional formats to choose: MCTQ or FAKELY_QUANT.

+++++++++++++++++++++++++++
MCTQ Quantization Format
+++++++++++++++++++++++++++

By default, `mct.exporter.pytorch_export_model` will export the quantized pytorch model to
an ONNX model with custom quantizers from mct_quantizers module.


.. code-block:: python

    # Path of exported model
    onnx_file_path = 'model_format_onnx_mctq.onnx'

    # Export ONNX model with mctq quantizers.
    mct.exporter.pytorch_export_model(model=quantized_exportable_model,
                                      save_model_path=onnx_file_path,
                                      repr_dataset=representative_data_gen)

Notice that the model has the same size as the quantized exportable model as weights data types are float.

+++++++++++++++++++++++++++
ONNX opset version
+++++++++++++++++++++++++++

By default, the used ONNX opset version is 15, but this can be changed using `onnx_opset_version`:

.. code-block:: python

    # Export ONNX model with mctq quantizers.
    mct.exporter.pytorch_export_model(model=quantized_exportable_model,
                                      save_model_path=onnx_file_path,
                                      repr_dataset=representative_data_gen,
                                      onnx_opset_version=16)

|

++++++++++++++++++++++++++++++++++++
Use exported model for inference
++++++++++++++++++++++++++++++++++++

To load and infer using the exported model, which was exported to an ONNX file in MCTQ format, we will use `mct_quantizers` method `get_ort_session_options` during onnxruntime session creation. **Notice**, inference on models that are exported in this format are slowly and suffers from longer latency. However, inference of these models on IMX500 will not suffer from this issue.

.. code-block:: python

    import mct_quantizers as mctq
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_file_path,
                                mctq.get_ort_session_options(),
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    _input_data = next(representative_data_gen())[0].astype(np.float32)
    _model_output_name = sess.get_outputs()[0].name
    _model_input_name = sess.get_inputs()[0].name

    # Run inference
    predictions = sess.run([_model_output_name], {_model_input_name: _input_data})

|

