import math
import model_compression_toolkit as mct
import logging
from integrations.common.consts import NUM_REPRESENTATIVE_IMAGES, BATCH_SIZE, REPRESENTATIVE_DATASET_FOLDER, \
    TARGET_PLATFORM_NAME, TARGET_PLATFORM_VERSION


def get_tpc(target_platform_name, target_platform_version):
    return mct.get_target_platform_capabilities('pytorch', target_platform_name, target_platform_version)


def quantize(model, get_representative_dataset, tpc, args):
    n_iter = math.ceil(int(args[NUM_REPRESENTATIVE_IMAGES]) // int(args[BATCH_SIZE]))
    logging.info(f"Running MCT... number of representative images: {args[REPRESENTATIVE_DATASET_FOLDER]}, number of calibration iters: {n_iter}")

    representative_data_gen = get_representative_dataset(
        representative_dataset_folder=args[REPRESENTATIVE_DATASET_FOLDER],
        n_iter=n_iter,
        batch_size=int(args[BATCH_SIZE])
    )

    quantized_model, quantization_info = \
        mct.ptq.pytorch_post_training_quantization_experimental(model,
                                                                representative_data_gen=representative_data_gen,
                                                                target_platform_capabilities=tpc)

    return quantized_model, quantization_info