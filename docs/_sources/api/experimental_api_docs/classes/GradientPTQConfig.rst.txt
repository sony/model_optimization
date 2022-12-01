:orphan:

.. _ug-GradientPTQConfig:

=================================
GradientPTQConfigV2 Class
=================================

**The following API can be used to create a GradientPTQConfigV2 instance which can be used for post training quantization using knowledge distillation from a teacher (float model) to a student (the quantized model). This is experimental and subject to future changes.**

.. autoclass:: model_compression_toolkit.GradientPTQConfigV2
    :members:


=================================
GradientPTQConfig Class
=================================

.. note:: GradientPTQConfig will be removed in future releases. Using GradientPTQConfigV2 is recommended.`

**The following API can be used to create a GradientPTQConfig instance which can be used for post training quantization using knowledge distillation from a teacher (float Keras model) to a student (the quantized Keras model)**

.. autoclass:: model_compression_toolkit.GradientPTQConfig
    :members:
