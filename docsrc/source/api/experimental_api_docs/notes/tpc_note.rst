
.. note::
   For now, fusing operators information from :class:`~model_compression_toolkit.target_platform.TargetPlatformModel`
   is ignored during the optimization process (fusing still occurs but using an existing mechanism).
   Also, parts of :class:`~model_compression_toolkit.target_platform.OpQuantizationConfig` is ignored (currently,
   the quantizer type, number of bits, and quantization enable/disable information is used during the
   optimization process).

   - MCT will use more information from :class:`~model_compression_toolkit.target_platform.TargetPlatformModel`, in the future.

