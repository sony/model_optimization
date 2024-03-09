:orphan:

.. _ug-api_docs:


=========
API Docs
=========

**Init module for MCT API.**

.. code-block:: python

   import model_compression_toolkit as mct

|


Functions
=========
- :ref:`pytorch_post_training_quantization<ug-pytorch_post_training_quantization>`: A function to use for post training quantization of PyTorch models.
- :ref:`keras_post_training_quantization<ug-keras_post_training_quantization>`: A function to use for post training quantization of Keras models.

- :ref:`keras_gradient_post_training_quantization<ug-keras_gradient_post_training_quantization>`: A function to use for gradient-based post training quantization of Keras models.
- :ref:`get_keras_gptq_config<ug-get_keras_gptq_config>`: A function to create a GradientPTQConfig instance to use for Keras models when using GPTQ.

- :ref:`pytorch_gradient_post_training_quantization<ug-pytorch_gradient_post_training_quantization>`: A function to use for gradient-based post training quantization of Pytorch models.
- :ref:`get_pytorch_gptq_config<ug-get_pytorch_gptq_config>`: A function to create a GradientPTQConfig instance to use for Pytorch models when using GPTQ.

- :ref:`keras_quantization_aware_training_init<ug-keras_quantization_aware_training_init_experimental>`: A function to use for preparing a model for Quantization Aware Training (Experimental)
- :ref:`keras_quantization_aware_training_finalize<ug-keras_quantization_aware_training_finalize_experimental>`: A function to finalize a model after Quantization Aware Training to a model without QuantizeWrappers (Experimental)

- :ref:`keras_data_generation_experimental<ug-keras_data_generation_experimental>`: A function to generate data for a Keras model (experimental).
- :ref:`get_keras_data_generation_config<ug-get_keras_data_generation_config>`: A function to generate a DataGenerationConfig for Tensorflow data generation(experimental).

- :ref:`pytorch_data_generation_experimental<ug-pytorch_data_generation_experimental>`: A function to generate data for a Pytorch model (experimental).
- :ref:`get_pytorch_data_generation_config<ug-get_pytorch_data_generation_config>`: A function to load a DataGenerationConfig for Pytorch data generation (experimental).

- :ref:`keras_pruning_experimental<ug-keras_pruning_experimental>`: A function to apply structured pruning for Keras models (experimental).
- :ref:`pytorch_pruning_experimental<ug-pytorch_pruning_experimental>`: A function to apply structured pruning for Pytorch models (experimental).

- :ref:`keras_kpi_data<ug-keras_kpi_data>`: A function to compute KPI data that can be used to calculate the desired target KPI for mixed-precision quantization of Keras models.
- :ref:`pytorch_kpi_data<ug-pytorch_kpi_data>`: A function to compute KPI data that can be used to calculate the desired target KPI for mixed-precision quantization of PyTorch models.

- :ref:`get_target_platform_capabilities<ug-get_target_platform_capabilities>`: A function to get a target platform model for Tensorflow and Pytorch.
- :ref:`keras_load_quantized_model<ug-keras_load_quantized_model>`: A function to load a quantized keras model.


Modules
=========
- :ref:`core_config<ug-core_config>`: Module to contain configurations of the optimization process.
- :ref:`quantization_config<ug-quantization_config>`: Module to configure the quantization process.
- :ref:`mixed_precision_quantization_config<ug-mixed_precision_quantization_config_v2>`: Module to configure the quantization process when using mixed-precision PTQ.
- :ref:`debug_config<ug-debug_config>`: Module to configure options for debugging the optimization process.
- :ref:`target_platform<ug-target_platform>`: Module to create and model hardware-related settings to optimize the model according to, by the hardware the optimized model will use during inference.
- :ref:`qat_config<ug-qat_config>`: Module to create quantization configuration for Quantization-aware Training.
- :ref:`exporter<ug-exporter>`: Module that enables to export a quantized model in different serialization formats.
- :ref:`trainable_infrastructure<ug-trainable_infrastructure>`: Module that contains quantization abstraction and quantizers for hardware-oriented model optimization tools.

Classes
=========
- :ref:`GradientPTQConfig<ug-GradientPTQConfig>`: Class to configure GradientPTQ options for gradient based post training quantization.
- :ref:`FolderImageLoader<ug-FolderImageLoader>`: Class to use an images directory as a representative dataset.
- :ref:`FrameworkInfo<ug-FrameworkInfo>`: Class to wrap framework information to be used by MCT when optimizing models.
- :ref:`PruningConfig<ug-PruningConfig>`: PruningConfig
- :ref:`PruningInfo<ug-PruningInfo>`: PruningInfo


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note:: This documentation is auto-generated using Sphinx

