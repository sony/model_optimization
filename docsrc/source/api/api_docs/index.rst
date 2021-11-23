=========
API Docs
=========

**Init module for MCT API.**

.. literalinclude:: ../../../../tutorials/example_keras_mobilenet.py
    :language: python
    :lines: 16

Functions
=========
- :ref:`keras_post_training_quantization<ug-keras_post_training_quantization>`: Function to use for post training quantization of Keras models.
- :ref:`keras_post_training_quantization_mixed_precision<ug-keras_post_training_quantization_mixed_precision>`: Function to use for mixed-precision post training quantization of Keras models (experimental).

Modules
=========
- :ref:`quantization_config<ug-quantization_config>`: Module to configure the quantization process.
- :ref:`mixed_precision_quantization_config<ug-mixed_precision_quantization_config>`: Module to configure the quantization process when using mixed-precision PTQ.
- :ref:`network_editor<ug-network_editor>`: Module to edit your model during the quantization process.

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

