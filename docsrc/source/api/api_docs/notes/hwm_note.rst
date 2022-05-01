
.. note::
   For now, the **only** information from :class:`~model_compression_toolkit.target_platform.HardwareModel`
   that MCT uses are the values of :class:`~model_compression_toolkit.target_platform.QuantizationMethod`
   (for weights and activations) from the default :class:`~model_compression_toolkit.target_platform.QuantizationConfigOptions` that is
   set to the :class:`~model_compression_toolkit.target_platform.HardwareModel`.

   - MCT will use more information from :class:`~model_compression_toolkit.target_platform.HardwareModel` gradually, in the future.

