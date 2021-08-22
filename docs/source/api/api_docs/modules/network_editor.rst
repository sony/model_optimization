:orphan:

.. _ug-network_editor:


=================================
network_editor Module
=================================

**The model can be edited by a list of EditRules to apply on nodes in a graph that represents the model during the model quantization. Each EditRule is a tuple of a filter and an action, where we apply the action on each node the filter matches**

EditRule
==========
.. autoclass:: network_optimization_package.network_editor.EditRule

Filters
==========

.. autoclass:: network_optimization_package.network_editor.NodeTypeFilter

|

.. autoclass:: network_optimization_package.network_editor.NodeNameFilter

|

.. autoclass:: network_optimization_package.network_editor.NodeNameScopeFilter


Actions
==========

.. autoclass:: network_optimization_package.network_editor.ChangeWeightsQuantConfigAttr

|

.. autoclass:: network_optimization_package.network_editor.ChangeActivationQuantConfigAttr


