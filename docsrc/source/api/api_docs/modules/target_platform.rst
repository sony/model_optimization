:orphan:

.. _ug-target_platform:


=================================
target_platform Module
=================================

MCT can be configured to quantize and optimize models for different hardware settings.
For example, when using qnnpack backend for Pytorch model inference, Pytorch `quantization
configuration <https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/qconfig.py#L199>`_
uses `per-tensor weights quantization <https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/observer.py#L1429>`_
for Conv2d, while when using tflite modeling, Tensorflow uses `per-channel weights quantization for
Conv2D <https://www.tensorflow.org/lite/performance/quantization_spec#per-axis_vs_per-tensor>`_.

This can be addressed in MCT by using the target_platform module, that can configure different
parameters that are hardware-related, and the optimization process will use this to optimize the model accordingly.
Models for IMX500, TFLite and qnnpack can be observed `here <https://github.com/sony/model_optimization/tree/main/model_compression_toolkit/target_platform_capabilities>`_, and can be used using :ref:`get_target_platform_capabilities function<ug-get_target_platform_capabilities>`.

|

.. include:: ../notes/tpc_note.rst

|

The object MCT should get called TargetPlatformCapabilities (or shortly TPC).
This diagram demonstrates the main components:

.. image:: ../../../../images/tpc.jpg
  :scale: 80%

Now, we will detail about the different components.



QuantizationMethod
==========================
.. autoclass:: model_compression_toolkit.target_platform.QuantizationMethod



OpQuantizationConfig
======================
.. autoclass:: model_compression_toolkit.target_platform.OpQuantizationConfig



AttributeQuantizationConfig
============================
.. autoclass:: model_compression_toolkit.target_platform.AttributeQuantizationConfig


QuantizationConfigOptions
============================
.. autoclass:: model_compression_toolkit.target_platform.QuantizationConfigOptions


TargetPlatformModel
=======================
.. autoclass:: model_compression_toolkit.target_platform.TargetPlatformModel


OperatorsSet
================
.. autoclass:: model_compression_toolkit.target_platform.OperatorsSet



Fusing
==============
.. autoclass:: model_compression_toolkit.target_platform.Fusing



OperatorSetConcat
====================
.. autoclass:: model_compression_toolkit.target_platform.OperatorSetConcat


OperationsToLayers
=====================
.. autoclass:: model_compression_toolkit.target_platform.OperationsToLayers


OperationsSetToLayers
=========================
.. autoclass:: model_compression_toolkit.target_platform.OperationsSetToLayers


LayerFilterParams
=========================
.. autoclass:: model_compression_toolkit.target_platform.LayerFilterParams

More filters and usage examples are detailed :ref:`here<ug-layer_filters>`.


TargetPlatformCapabilities
=============================
.. autoclass:: model_compression_toolkit.target_platform.TargetPlatformCapabilities



