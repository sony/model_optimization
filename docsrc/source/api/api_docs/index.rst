:orphan:

.. _ug-api_docs:


=========
API Docs
=========

**Init module for MCT API.**

.. code-block:: python

   import model_compression_toolkit as mct

|


ptq
====

- :ref:`pytorch_post_training_quantization<ug-pytorch_post_training_quantization>`: A function to use for post training quantization of PyTorch models.
- :ref:`keras_post_training_quantization<ug-keras_post_training_quantization>`: A function to use for post training quantization of Keras models.

gptq
====

- :ref:`pytorch_gradient_post_training_quantization<ug-pytorch_gradient_post_training_quantization>`: A function to use for gradient-based post training quantization of Pytorch models.
- :ref:`get_pytorch_gptq_config<ug-get_pytorch_gptq_config>`: A function to create a GradientPTQConfig instance to use for Pytorch models when using GPTQ.

- :ref:`keras_gradient_post_training_quantization<ug-keras_gradient_post_training_quantization>`: A function to use for gradient-based post training quantization of Keras models.
- :ref:`get_keras_gptq_config<ug-get_keras_gptq_config>`: A function to create a GradientPTQConfig instance to use for Keras models when using GPTQ.

- :ref:`GradientPTQConfig<ug-GradientPTQConfig>`: Class to configure GradientPTQ options for gradient based post training quantization.

qat
====

- :ref:`pytorch_quantization_aware_training_init_experimental<ug-pytorch_quantization_aware_training_init_experimental>`: A function to use for preparing a Pytorch model for Quantization Aware Training (experimental).
- :ref:`pytorch_quantization_aware_training_finalize_experimental<ug-pytorch_quantization_aware_training_finalize_experimental>`: A function to finalize a Pytorch model after Quantization Aware Training to a model without QuantizeWrappers (experimental).
- :ref:`keras_quantization_aware_training_init_experimental<ug-keras_quantization_aware_training_init_experimental>`: A function to use for preparing a Keras model for Quantization Aware Training (experimental).
- :ref:`keras_quantization_aware_training_finalize_experimental<ug-keras_quantization_aware_training_finalize_experimental>`: A function to finalize a Keras model after Quantization Aware Training to a model without QuantizeWrappers (experimental).
- :ref:`qat_config<ug-qat_config>`: Module to create quantization configuration for Quantization-aware Training (experimental).

core
=====

- :ref:`CoreConfig<ug-CoreConfig>`: Module to contain configurations of the entire optimization process.
- :ref:`QuantizationConfig<ug-QuantizationConfig>`: Module to configure the quantization process.
- :ref:`QuantizationErrorMethod<ug-QuantizationErrorMethod>`: Select a method for quantization parameters' selection.
- :ref:`MixedPrecisionQuantizationConfig<ug-MixedPrecisionQuantizationConfig>`: Module to configure the quantization process when using mixed-precision PTQ.
- :ref:`BitWidthConfig<ug-BitWidthConfig>`: Module to configure the bit-width manually.
- :ref:`ResourceUtilization<ug-ResourceUtilization>`: Module to configure resources to use when searching for a configuration for the optimized model.
- :ref:`network_editor<ug-network_editor>`: Module to modify the optimization process for troubleshooting.
- :ref:`pytorch_resource_utilization_data<ug-pytorch_resource_utilization_data>`: A function to compute Resource Utilization data that can be used to calculate the desired target resource utilization for PyTorch models.
- :ref:`keras_resource_utilization_data<ug-keras_resource_utilization_data>`: A function to compute Resource Utilization data that can be used to calculate the desired target resource utilization for Keras models.


data_generation
=================

- :ref:`pytorch_data_generation_experimental<ug-pytorch_data_generation_experimental>`: A function to generate data for a Pytorch model (experimental).
- :ref:`get_pytorch_data_generation_config<ug-get_pytorch_data_generation_config>`: A function to load a DataGenerationConfig for Pytorch data generation (experimental).
- :ref:`keras_data_generation_experimental<ug-keras_data_generation_experimental>`: A function to generate data for a Keras model (experimental).
- :ref:`get_keras_data_generation_config<ug-get_keras_data_generation_config>`: A function to generate a DataGenerationConfig for Tensorflow data generation (experimental).
- :ref:`DataGenerationConfig<ug-DataGenerationConfig>`: A configuration class for the data generation process (experimental).


pruning
===========

- :ref:`pytorch_pruning_experimental<ug-pytorch_pruning_experimental>`: A function to apply structured pruning for Pytorch models (experimental).
- :ref:`keras_pruning_experimental<ug-keras_pruning_experimental>`: A function to apply structured pruning for Keras models (experimental).

- :ref:`PruningConfig<ug-PruningConfig>`: Configuration for the pruning process (experimental).
- :ref:`PruningInfo<ug-PruningInfo>`: Information about the pruned model such as pruned channel indices, etc. (experimental).

xquant
===========

- :ref:`xquant_report_pytorch_experimental<ug-xquant_report_pytorch_experimental>`: A function to generate an explainable quantization report for a quantized Pytorch model (experimental).
- :ref:`xquant_report_keras_experimental<ug-xquant_report_keras_experimental>`: A function to generate an explainable quantization report for a quantized Keras model (experimental).

- :ref:`XQuantConfig<ug-XQuantConfig>`: Configuration for the XQuant report (experimental).

exporter
=========

- :ref:`exporter<ug-exporter>`: Module that enables to export a quantized model in different serialization formats.


trainable_infrastructure
=========================

- :ref:`trainable_infrastructure<ug-trainable_infrastructure>`: Module that contains quantization abstraction and quantizers for hardware-oriented model optimization tools.



set_log_folder
================
- :ref:`set_log_folder<ug-set_log_folder>`: Function to set the logger path directory and to enable logging.

keras_load_quantized_model
============================
- :ref:`keras_load_quantized_model<ug-keras_load_quantized_model>`: A function to load a quantized keras model.


target_platform_capabilities
==============================
- :ref:`target_platform_capabilities<ug-target_platform_capabilities>`: Module to create and model hardware-related settings to optimize the model according to, by the hardware the optimized model will use during inference.
- :ref:`get_target_platform_capabilities<ug-get_target_platform_capabilities>`: A function to get a target platform model for Tensorflow and Pytorch.
- :ref:`DefaultDict<ug-DefaultDict>`: Util class for creating a TargetPlatformCapabilities.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note:: This documentation is auto-generated using Sphinx

