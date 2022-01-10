:orphan:

.. _ug-quickstart-keras:

==========================
MCT Quickstart Guideline
==========================

Here is an example of a code that shows how to use MCT with Keras models.


Import MCT and MobileNetV1 from Keras applications:

.. literalinclude:: ../../../tutorials/example_keras_mobilenet.py
    :language: python
    :lines: 16-17

|

Data preprocessing functions:

.. literalinclude:: ../../../tutorials/example_keras_mobilenet.py
    :language: python
    :lines: 27-48

|

Initialize data loader:

.. literalinclude:: ../../../tutorials/example_keras_mobilenet.py
    :language: python
    :lines: 53-75

|

Run Post Training Quantization:

.. literalinclude:: ../../../tutorials/example_keras_mobilenet.py
    :language: python
    :lines: 77-82

