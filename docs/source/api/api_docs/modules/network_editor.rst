:orphan:

.. _ug-network_editor:


=================================
network_editor Module
=================================

**The model can be edited by a list of EditRules to apply on nodes in a graph that represents the model during the model quantization. Each EditRule is a tuple of a filter and an action, where we apply the action on each node the filter matches**

EditRule
==========
.. autoclass:: sony_model_optimization_package.network_editor.EditRule

Filters
==========

.. autoclass:: sony_model_optimization_package.network_editor.NodeTypeFilter

|

.. autoclass:: sony_model_optimization_package.network_editor.NodeNameFilter

|

.. autoclass:: sony_model_optimization_package.network_editor.NodeNameScopeFilter


Actions
==========

.. autoclass:: sony_model_optimization_package.network_editor.ChangeWeightsQuantConfigAttr

|

.. autoclass:: sony_model_optimization_package.network_editor.ChangeActivationQuantConfigAttr


