import math
import model_compression_toolkit as mct


def quant(ModelSrc, args):
    n_iter = math.ceil(args.n_images // args.batch_size)
    model = ModelSrc.get_model()

    representative_data_gen = ModelSrc.get_representative_dataset(
        representative_dataset_folder=args.representative_dataset_folder,
        n_iter=n_iter,
        batch_size=args.batch_size,
        n_images=args.n_images,
        image_size=args.image_size,
        preprocessing=None,
        seed=args.random_seed)


    quantized_model, quantization_info = \
        mct.ptq.pytorch_post_training_quantization_experimental(model,
                                                                representative_data_gen=representative_data_gen)


    # conversion of MCT model to type of the original model
    # model_for_eval = func(float_model, quantization_config, representative_data_gen)

    return quantized_model