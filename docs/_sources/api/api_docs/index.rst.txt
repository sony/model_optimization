:orphan:

.. _ug-api_docs:


=========
API Docs
=========

.. note:: This API will be removed in future releases. Please switch to the :ref:`new API<ug-experimental_api_docs>`

**Init module for MCT API.**

.. literalinclude:: ../../../../tutorials/example_keras_mobilenet.py
    :language: python
    :lines: 16

Functions
=========
- :ref:`pytorch_post_training_quantization<ug-pytorch_post_training_quantization>`: Function to use for post training quantization of Pytorch models.
- :ref:`pytorch_post_training_quantization_mixed_precision<ug-pytorch_post_training_quantization_mixed_precision>`: Function to use for mixed-precision post training quantization of Pytorch models (experimental).
- :ref:`keras_post_training_quantization<ug-keras_post_training_quantization>`: Function to use for post training quantization of Keras models.
- :ref:`keras_post_training_quantization_mixed_precision<ug-keras_post_training_quantization_mixed_precision>`: Function to use for mixed-precision post training quantization of Keras models (experimental).
- :ref:`get_keras_gptq_config<ug-get_keras_gptq_config>`: Function to create a GradientPTQConfig instance to use for Keras models when using GPTQ (experimental).
- :ref:`get_target_platform_capabilities<ug-get_target_platform_capabilities>`: Function to get a target platform model for Tensorflow and Pytorch.
- :ref:`keras_kpi_data<ug-keras_kpi_data>`: Function to compute KPI data that can be used to calculate the desired target KPI for mixed-precision quantization of Keras models.
- :ref:`pytorch_kpi_data<ug-pytorch_kpi_data>`: Function to compute KPI data that can be used to calculate the desired target KPI for mixed-precision quantization of PyTorch models.


Modules
=========
- :ref:`quantization_config<ug-quantization_config>`: Module to configure the quantization process.
- :ref:`mixed_precision_quantization_config<ug-mixed_precision_quantization_config>`: Module to configure the quantization process when using mixed-precision PTQ.
- :ref:`network_editor<ug-network_editor>`: Module to edit your model during the quantization process.
- :ref:`target_platform<ug-target_platform>`: Module to create and model hardware-related settings to optimize the model according to, by the hardware the optimized model will use during inference.

Classes
=========
- :ref:`GradientPTQConfig<ug-GradientPTQConfig>`: Class to configure GradientPTQC options for gradient based post training quantization.
- :ref:`FolderImageLoader<ug-FolderImageLoader>`: Class to use an images directory as a representative dataset.
- :ref:`FrameworkInfo<ug-FrameworkInfo>`: Class to wrap framework information to be used by MCT when optimizing models.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note:: This documentation is auto-generated using Sphinx

