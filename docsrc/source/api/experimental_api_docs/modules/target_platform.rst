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
Models for TFLite and qnnpack can be observed `here <https://github.com/sony/model_optimization/tree/main/model_compression_toolkit/core/tpc_models>`_, and can be used using :ref:`get_target_platform_capabilities function<ug-get_target_platform_capabilities>`.

|

.. include:: ../notes/tpc_note.rst

|

The object MCT should get called TargetPlatformCapabilities (or shortly TPC).
This diagram demonstrates the main components:

.. image:: ../../../../images/tpc.jpg
  :scale: 80%

Now, we will explain about each component with examples.

The first part is configuring the quantization method for both wights and activations of an operator.
Several methods can be used using QuantizationMethod API:


QuantizationMethod
==========================
Select a method to use during quantization:

.. autoclass:: model_compression_toolkit.target_platform.QuantizationMethod


|


Using a quantization method (or methods, if the weights and activations of an operator are quantized differently)
Quantization configuration of different operators can be created using OpQuantizationConfig:


OpQuantizationConfig
======================
.. autoclass:: model_compression_toolkit.target_platform.OpQuantizationConfig

|

If, for example, we would like to quantize an operator's weights with 8 bits (and per-channel), its activations
with 8 bits, and the quantization thresholds (for both weights and activations) must be power-of-two,
we can create the OpQuantizationConfig:

.. code-block:: python

   op_qc_8bit = OpQuantizationConfig(
       activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=8,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True
    )

|

We will demonstrate later how to attach this OpQuantizationConfig to a specific operator.

If an operator can be quantized in different ways (the simplest example is mixed-precision quantization),
one can create a QuantizationConfigOptions instance to represent a set of possible quantization
configuration options for an operator:


QuantizationConfigOptions
============================
.. autoclass:: model_compression_toolkit.target_platform.QuantizationConfigOptions

If a QuantizationConfigOptions is created with more than
one OpQuantizationConfig option, a base_config must be passed to the QuantizationConfigOptions
in order to support the model when MCT optimizes the model in no mixed-precision manner.

For example, we would like to quantize an operator's weights with either 2, 4 or 8 bits (and in
case we would like to use MCT non mixed-precision functions, we would like to quantize the operator
using 8 bits). For this we can create new OpQuantizationConfigs based on previously created
OpQuantizationConfigs, and gather them under a single QuantizationConfigOptions instance:

.. code-block:: python

    # To quantize a model using mixed-precision, create a QuantizationConfigOptions with more
    # than one QuantizationConfig.
    # In this example, we aim to quantize some operations' weights using 2, 4 or 8 bits.
    op_qc_4bit = op_qc_8bit.clone_and_edit(weights_n_bits=4)
    op_qc_2bit = op_qc_8bit.clone_and_edit(weights_n_bits=2)
    mixed_precision_configuration_options = QuantizationConfigOptions([op_qc_8bit,
                                                                       op_qc_4bit,
                                                                       op_qc_2bit],
                                                                       base_config=op_qc_8bit)

|



The main class to define the hardware-related properties, is called TargetPlatformModel. Using a TargetPlatformModel
object we can create operator sets, configure how these operators sets will be quantized,
group operators by common properties and configure patterns of operators to fuse:


TargetPlatformModel
=======================
.. autoclass:: model_compression_toolkit.target_platform.TargetPlatformModel


A default QuantizationConfigOptions (containing a single OpQuantizationConfig) must be passed
when instancing a TargetPlatformModel object. It comes to use when MCT needs to optimize
an operator that is not defined explicitly in the TargetPlatformModel. In this case, the OpQuantizationConfig
in the default QuantizationConfigOptions will guide MCT how this operator should be optimized. For example:

.. code-block:: python

    # Create a QuantizationConfigOptions with a single OpQuantizationConfig to use as
    # a default configuration options.
    default_configuration_options = QuantizationConfigOptions([op_qc_8bit])

    # Create a TargetPlatformModel and set its default quantization config.
    # This default configuration will be used for all operations
    # unless specified otherwise:
    my_model = TargetPlatformModel(default_configuration_options, name='my_model')

|

Then, we can start defining the model by creating OperatorsSets:

OperatorsSet
================
.. autoclass:: model_compression_toolkit.target_platform.OperatorsSet

An OperatorsSet gathers group of operators that are labeled by a unique name and can be attached to a
QuantizationConfigOptions (so MCT will use these options to optimize operators from this set).
For example, if FullyConnected can be quantized using 2, 4, or 8 bits, we can create the next
OperatorsSet using the previously created mixed_precision_configuration_options:

.. code-block:: python

    # Define operators set named "FullyConnected" and attach
    # mixed_precision_configuration_options as its QuantizationConfigOptions:
    fc_opset = OperatorsSet("FullyConnected", mixed_precision_configuration_options)

|

The QuantizationConfigOptions is optional. An OperatorsSet can be also created
without any attached QuantizationConfigOptions. Operators in this kind of OperatorsSets
are attached implicitly to the default QuantizationConfigOptions of the TargetPlatformModel
they are part of:

.. code-block:: python

    # Define operators set named "Relu" and do not attach
    # it any QuantizationConfigOptions:
    relu_opset = OperatorsSet("Relu")

|

Another component of a TargetPlatformModel is Fusing. Fusing defines a list
of operators that should be combined and treated as a single operator, hence no
quantization is applied between them when they appear in a model:


Fusing
==============
.. autoclass:: model_compression_toolkit.target_platform.Fusing

For example, to fuse the previously created two OperatorsSets fc_opset and
relu_opset we can create the next Fusing:

.. code-block:: python

    # Combine multiple operators into a single operator to avoid quantization between
    # them. To do this we define fusing patterns using the OperatorsSets that were created.
    Fusing([fc_opset, relu_opset])

|

Notice that the list of opsets must contain at least two OperatorSets.
Also notice that sublist of the OperatorsSet list that is passed to the Fusing,
will not be fused, unless another Fusing is created for that. For example,
if a model is defined to fuse three sequenced operators [FullyConnected, Relu, Add]:

.. code-block:: python

    # In addition to the OperatorsSets we created, create new OperatorsSets for "add" ops:
    add_opset = OperatorsSet("Add")

    # Fuse sequences of operators:
    Fusing([fc_opset, relu_opset, add_opset])

|

and the pre-trained model that MCT optimizes has a sequence of [fc_opset, relu_opset]
where the next operator is not an add_opset, the two operators [fc_opset, relu_opset]
will not be fused as the only defined fusing pattern is of the three OperatorsSets
[fc_opset, relu_opset, add_opset]. In order to fuse sequences of [fc_opset, relu_opset]
as well, a new Fusing should be defined:

.. code-block:: python

    # Fuse sequences of the three listed operators:
    Fusing([fc_opset, relu_opset, add_opset])

    # In addition, fuse sequences of the two listed operators:
    Fusing([fc_opset, relu_opset])

Now, if MCT encounters a sequence of [fc_opset, relu_opset] they will be fused regardless the following operator.
Sequences of [fc_opset, relu_opset, add_opset] will be fused as well, and
the new Fusing of [fc_opset, relu_opset] will not affect them (but will affect patterns
of [fc_opset, relu_opset], of course).

When multiple operators should be fused in a similar way, an OperatorSetConcat can be used:

OperatorSetConcat
====================
.. autoclass:: model_compression_toolkit.target_platform.OperatorSetConcat


OperatorSetConcat gathers multiple OperatorsSet and can be specified in a fusing operators list.
If, for example, we want to fuse the patterns [fc_opset, add_opset] and [fc_opset, relu_opset],
we can either create two separate Fusing objects as was demonstrated above, or an OperatorSetConcat
can be used as follows:

.. code-block:: python

    # Concatenate two OpseratorsSet objects to be treated similarly when fused:
    activations_after_fc_to_fuse = OperatorSetConcat(relu_opset, add_opset)

    # Create a fusing pattern using OperatorSetConcat. This is equivalent to define two
    # separate fusing patterns of: [fc_opset, relu_opset], [fc_opset, add_opset]
    Fusing([fc_opset, activations_after_fc_to_fuse])

|


TargetPlatformModel Code Example
===================================

.. literalinclude:: ../../../../../model_compression_toolkit/core/tpc_models/default_tpc/v3/tp_model.py
    :language: python
    :lines: 15-158

|

After modeling the hardware MCT should optimize according to, this hardware model needs to be
attached to a specific framework information in order to associate the operators that are defined in
hardware model to layers in different representations of a framework.
For example, if we created an OperatorsSet for "Add" operator, in Tensorflow this operator
can be used by two different layers: keras.layers.Add, tf.add.
To attach a list of framework's layers to an OperatorsSet that is defined in the TargetPlatformModel,
an OperationsSetToLayers can be used:

OperationsSetToLayers
=========================
.. autoclass:: model_compression_toolkit.target_platform.OperationsSetToLayers

Using OperationsSetToLayers we can associate an OperatorsSet label to a list of framework's layers:

.. code-block:: python

    import tensorflow as tf
    from keras.layers import Add
    OperationsSetToLayers("Add", [tf.add, Add])

|

This way, when MCT quantizes one of the layers tf.add or keras.layers.Add, it uses the QuantizationConfigOptions
that is associated with the OperatorsSet that was labeled "Add" to optimize the layer.

There are cases where an operator can be represented using a layer but it must have a specific configuration.

For example, in case the optimization should be different for bounded ReLU and unbounded ReLU, two OperatorSets
can be created, and the layers that will be attached to each OperatorSet will have to be filtered.
For that, LayerFilterParams can be used:

LayerFilterParams
=========================
.. autoclass:: model_compression_toolkit.target_platform.LayerFilterParams


LayerFilterParams wraps a layer with several conditions and key-value pairs
and can check whether a layer matches the layer, conditions and key-value pairs.
If for example a distinguish need to be made between bounded-ReLU and unbounded-ReLU in Tensorflow
the next LayerFilterParams can be created:

.. code-block:: python

    from keras.layers import ReLU

    # Create a LayerFilterParams that matches ReLU layers that have an attribute 'max_value'
    # and it is None
    unbounded_relu_filter = LayerFilterParams(ReLU, max_value=None)

    # Create a LayerFilterParams that matches ReLU layers that have an attribute 'max_value'
    # and it is not None
    unbounded_relu_filter = LayerFilterParams(ReLU, NotEq('max_value', None))

|

In this example, we used NotEq which is a way to filter layers with attributes that has
a value different than the value that was passed (in this case - None). More filters can be created
and passed to the LayerFilterParams in order to create more detailed filter.
More filters and usage examples are detailed :ref:`here<ug-layer_filters>`.

These LayerFilterParams instances can now be attached to OperatorsSets in the TargetPlatformModel
using OperationsSetToLayers just like any other layers:

.. code-block:: python

    import tensorflow as tf
    from keras.layers import ReLU, Activation

    OperationsSetToLayers("ReLU", [tf.nn.relu,
                                   tf.nn.relu6,
                                   LayerFilterParams(ReLU, negative_slope=0.0),
                                   LayerFilterParams(Activation, activation="relu")])

|

The mapping from OperatorsSets to layers' lists are part of a class called TargetPlatformCapabilities
which attaches the layers representations to OperatorsSets in a TargetPlatformModel instance:

TargetPlatformCapabilities
=============================
.. autoclass:: model_compression_toolkit.target_platform.TargetPlatformCapabilities


To create a TargetPlatformCapabilities, a TargetPlatformModel instance should be passed upon the
TargetPlatformCapabilities initialization. Then, OperationsSetToLayers can be created and attached
to the TargetPlatformCapabilities like in the following example:


TargetPlatformCapabilities Code Example
===========================================

.. literalinclude:: ../../../../../model_compression_toolkit/core/tpc_models/default_tpc/v3/tpc_keras.py
    :language: python
    :lines: 15-86




