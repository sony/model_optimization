 :orphan:

.. _ug-quickstart-keras:

==========================
SMOP Quickstart Guideline
==========================

Here is an example of a code that shows how to use SMOP with Keras models.


Import SMOP and MobileNetV1 from Keras applications:

.. literalinclude:: ../../../tutorials/example_keras_mobilenet.py
    :language: python
    :lines: 30-31

|

Data preprocessing functions:

.. literalinclude:: ../../../tutorials/example_keras_mobilenet.py
    :language: python
    :lines: 41-62

|

Initialize data loader:

.. literalinclude:: ../../../tutorials/example_keras_mobilenet.py
    :language: python
    :lines: 67-82

|

Run Post Training Quantization:

.. literalinclude:: ../../../tutorials/example_keras_mobilenet.py
    :language: python
    :lines: 84-89

