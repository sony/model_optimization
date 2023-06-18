:orphan:

.. _ug-exporter:


=================================
exporter Module
=================================

Allows to export a quantized model in the following serialization formats:

- TensorFlow models can be exported as Tensorflow models (.h5 extension) and TFLite models (.tflite extension).
- PyTorch models can be exported as torch script models and ONNX models (.onnx extension).

Also, allows to export quantized model in the following quantization formats:

- Fake Quant (where weights and activations are float fakely-quantized values)
- INT8 (where weights and activations are represented using 8bits integers)

For more details about the export formats and options, please refer to the project's GitHub `README file <https://github.com/sony/model_optimization/tree/main/model_compression_toolkit/exporter>`_.
Note that this feature is experimental and subject to future changes. If you have any questions or issues, please open an issue in this GitHub repository.


KerasExportSerializationFormat
================================
Select the serialization format for exporting a quantized Keras model.

.. autoclass:: model_compression_toolkit.exporter.KerasExportSerializationFormat


PytorchExportSerializationFormat
==================================
Select the serialization format for exporting a quantized Pytorch model.

.. autoclass:: model_compression_toolkit.exporter.PytorchExportSerializationFormat


keras_export_model
==========================
Allows to export a Keras model that was quantized via MCT.

.. autoclass:: model_compression_toolkit.exporter.keras_export_model

Here is an example for how to export a quantized Keras model in a TFLite fakly-quantized format:

.. code-block:: python

    from mct.target_platform_capabilities.tpc_models.default_tpc.latest import get_keras_tpc_latest
    from mct.exporter import KerasExportSerializationFormat

    # Path of exported model
    _, tflite_file_path = tempfile.mkstemp('.tflite')

    # Get TPC
    keras_tpc = get_keras_tpc_latest()

    # Use mode KerasExportSerializationFormat.TFLITE for tflite model and keras tpc for fakely-quantized weights
    # and activations
    mct.exporter.keras_export_model(model=quantized_exportable_model,
                                    save_model_path=tflite_file_path,
                                    target_platform_capabilities=keras_tpc,
                                    serialization_format=KerasExportSerializationFormat.TFLITE)


|

pytorch_export_model
==========================
Allows to export a Pytorch model that was quantized via MCT.

.. autoclass:: model_compression_toolkit.exporter.pytorch_export_model

Here is an example for how to export a quantized Pytorch model in a ONNX fakly-quantized format:

.. code-block:: python

    import tempfile

    from mct.target_platform_capabilities.tpc_models.default_tpc.latest import get_pytorch_tpc_latest
    from mct.exporter import PytorchExportSerializationFormat

    # Path of exported model
    _, onnx_file_path = tempfile.mkstemp('.onnx')

    # Get TPC
    pytorch_tpc = get_pytorch_tpc_latest()

    # Use mode PytorchExportSerializationFormat.ONNX for keras h5 model and default pytorch tpc for fakely-quantized weights
    # and activations
    mct.exporter.pytorch_export_model(model=quantized_exportable_model, save_model_path=onnx_file_path,
                                      repr_dataset=representative_data_gen, target_platform_capabilities=pytorch_tpc,
                                      serialization_format=PytorchExportSerializationFormat.ONNX)


|
