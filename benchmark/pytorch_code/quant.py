import math
import model_compression_toolkit as mct


def get_tpc():
    return mct.get_target_platform_capabilities('pytorch', 'default')


def quantize(model, get_representative_dataset, tpc, args):
    n_iter = math.ceil(int(args['num_representative_images']) // int(args['batch_size']))
    print(f"Running MCT... number of representative images: {args['num_representative_images']}, number of calibration iters: {n_iter}")

    representative_data_gen = get_representative_dataset(
        representative_dataset_folder=args['representative_dataset_folder'],
        n_iter=n_iter,
        batch_size=int(args['batch_size'])
    )


    quantized_model, quantization_info = \
        mct.ptq.pytorch_post_training_quantization_experimental(model,
                                                                representative_data_gen=representative_data_gen,
                                                                target_platform_capabilities=tpc)


    # conversion of MCT model to type of the original model
    # model_for_eval = func(float_model, quantization_config, representative_data_gen)

    return quantized_model, quantization_info