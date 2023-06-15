:orphan:

.. _ug-GradientPTQConfig:

=================================
GradientPTQConfigV2 Class
=================================

**The following API can be used to create a GradientPTQConfigV2 instance which can be used for post training quantization using knowledge distillation from a teacher (float model) to a student (the quantized model). This is experimental and subject to future changes.**

.. autoclass:: model_compression_toolkit.gptq.GradientPTQConfigV2
    :members:


=================================
GradientPTQConfig Class
=================================


**The following API can be used to create a GradientPTQConfig instance which can be used for post training quantization using knowledge distillation from a teacher (float Keras model) to a student (the quantized Keras model)**

.. autoclass:: model_compression_toolkit.gptq.GradientPTQConfig
    :members:

=================================
GPTQHessianWeightsConfig Class
=================================


**Configuration to use for computing the Hessian-based weights for GPTQ loss metric.**

.. autoclass:: model_compression_toolkit.gptq.GPTQHessianWeightsConfig
    :members:
