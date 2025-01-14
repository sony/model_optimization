:orphan:

.. _ug-trainable_infrastructure:


=================================
trainable_infrastructure Module
=================================

The trainable infrastructure is a module containing quantization abstraction and quantizers for hardware-oriented model optimization tools.
It provides the required abstraction for trainable quantization methods such as quantization-aware training.
It utilizes the Inferable Quantizers Infrastructure provided by the `MCT Quantizers <https://github.com/sony/mct_quantizers>`_ package, which proposes the required abstraction for emulating inference-time quantization.

When using a trainable quantizer, each layer with quantized weights is wrapped with a "Quantization Wrapper" object,
and each activation quantizer is being stored in an "Activation Quantization Holder" object.
Both components are provided by the MCT Quantizers package.

The quantizers in this module are built upon the "Inferable Quantizer" abstraction (from MCT Quantizers),
and define the "Trainable Quantizer" framework,
which contains learnable quantization parameters that can be optimized during training.

Now, we will explain how a trainable quantizer is built and used.
We start by explaining the basic building block of a trainable quantizer, and then explain how to initialize it using a configuration object.

BaseKerasTrainableQuantizer
==============================
This class is a base class for trainable Keras quantizers which validates provided quantization config and defines an abstract function which any quantizer needs to implement.
It adds to the base quantizer a get_config and from_config functions to enable loading and saving the keras model.

.. autoclass:: model_compression_toolkit.trainable_infrastructure.BaseKerasTrainableQuantizer

BasePytorchTrainableQuantizer
==============================
This class is a base class for trainable Pytorch quantizers which validates provided quantization config and defines an abstract function which any quantizer needs to implement.
It adds to the base quantizer a get_config and from_config functions to enable loading and saving the keras model.

.. autoclass:: model_compression_toolkit.trainable_infrastructure.BasePytorchTrainableQuantizer



TrainingMethod
================================
**Select a training method:**

.. autoclass:: model_compression_toolkit.trainable_infrastructure.TrainingMethod


TrainableQuantizerWeightsConfig
=================================
This configuration object contains the necessary attributes for configuring a weights trainable quantizer.

.. autoclass:: model_compression_toolkit.trainable_infrastructure.TrainableQuantizerWeightsConfig

For example, we can set a trainable weights quantizer with the following configuration:

.. code-block:: python

    from model_compression_toolkit.target_platform_capabilities.target_platform_capabilities import QuantizationMethod
    from model_compression_toolkit.constants import THRESHOLD, MIN_THRESHOLD

    TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod.SYMMETRIC,
                                               weights_n_bits=8,
                                               weights_quantization_params={THRESHOLD: 2.0},
                                               enable_weights_quantization=True,
                                               weights_channels_axis=3,
                                               weights_per_channel_threshold=True,
                                               min_threshold=MIN_THRESHOLD)


|

TrainableQuantizerActivationConfig
====================================
This configuration object contains the necessary attributes for configuring an activation trainable quantizer.

.. autoclass:: model_compression_toolkit.trainable_infrastructure.TrainableQuantizerActivationConfig

For example, we can set a trainable activation quantizer with the following configuration:

.. code-block:: python

    from model_compression_toolkit.target_platform_capabilities.target_platform_capabilities import QuantizationMethod
    from model_compression_toolkit.constants import THRESHOLD, MIN_THRESHOLD

    TrainableQuantizerActivationConfig(activation_quantization_method=QuantizationMethod.UNIFORM,
                                                  activation_n_bits=8,
                                                  activation_quantization_params=={THRESHOLD: 2.0},
                                                  enable_activation_quantization=True,
                                                  min_threshold=MIN_THRESHOLD)


|
