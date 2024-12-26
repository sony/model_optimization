from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformModel


def get_tpc_model(name: str, tp_model: TargetPlatformModel):
    # TODO: this is just to adjust for tests so maybe it can be moved there
    return tp_model