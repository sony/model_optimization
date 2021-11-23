:orphan:

.. _ug-network_editor:


=================================
network_editor Module
=================================

**The model can be edited by a list of EditRules to apply on nodes in a graph that represents the model during the model quantization. Each EditRule is a tuple of a filter and an action, where we apply the action on each node the filter matches**

EditRule
==========
.. autoclass:: model_compression_toolkit.network_editor.EditRule

Filters
==========

.. autoclass:: model_compression_toolkit.network_editor.NodeTypeFilter

|

.. autoclass:: model_compression_toolkit.network_editor.NodeNameFilter

|

.. autoclass:: model_compression_toolkit.network_editor.NodeNameScopeFilter


Actions
==========

.. autoclass:: model_compression_toolkit.network_editor.ChangeCandidatesWeightsQuantConfigAttr

|

.. autoclass:: model_compression_toolkit.network_editor.ChangeActivationQuantConfigAttr

|

.. autoclass:: model_compression_toolkit.network_editor.ChangeQuantizationParamFunction

|

.. autoclass:: model_compression_toolkit.network_editor.ChangeActivationQuantizationMethod

|

.. autoclass:: model_compression_toolkit.network_editor.ChangeFinalWeightsQuantizationMethod

|

.. autoclass:: model_compression_toolkit.network_editor.ChangeCandidtaesWeightsQuantizationMethod

