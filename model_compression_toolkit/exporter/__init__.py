from model_compression_toolkit.core.common.constants import FOUND_TF

if FOUND_TF:
    from model_compression_toolkit.exporter.fully_quantized.keras.builder.fully_quantized_model_builder import \
        get_fully_quantized_keras_model