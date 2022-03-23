
.. note::
   For now, the **only** information from :class:`~model_compression_toolkit.hardware_representation.HardwareModel`
   that MCT uses are the values of :class:`~model_compression_toolkit.hardware_representation.QuantizationMethod`
   (for weights and activations) from the default :class:`~model_compression_toolkit.hardware_representation.QuantizationConfigOptions` that is
   set to the :class:`~model_compression_toolkit.hardware_representation.HardwareModel`.

   - MCT will use more information from :class:`~model_compression_toolkit.hardware_representation.HardwareModel` gradually, in the future.

