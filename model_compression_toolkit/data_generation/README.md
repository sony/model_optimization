# Data Generation Library for Model Compression Toolkit (MCT)

The Data Generation Library for the Model Compression Toolkit (MCT) is a powerful Python package designed to facilitate the generation of synthetic data. This library leverages the statistics stored in the model's batch normalization (BatchNorm) layers to create synthetic data that faithfully represent the model's training data characteristics. This generated data is valuable for various compression tasks where representative data of the model's training set is needed, such as quantization and pruning.


## Key Features

- **Synthetic Data Generation**: Generate synthetic data based on the statistics from the model's BatchNorm layers.
- **Complementary to Compression Tasks**: Use the generated data to improve various compression tasks, including quantization and pruning of neural networks.
- **User-Friendly Interface**: The library provides easy-to-use functions, making data generation seamless and accessible to researchers, developers, and engineers.
- **Integrates with MCT**: The Data Generation Library seamlessly integrates with the Model Compression Toolkit, enhancing its capabilities for neural network optimization and deployment.

## Installation
To install the package, you can use pip and install the latest stable release of the Model Compression Toolkit:
```bash
pip install model-compression-toolkit
```

## Usage

```python
from model_compression_toolkit.data_generation import get_pytorch_data_generation_config, pytorch_data_generation_experimental

# Set the configuration parameters for data generation
data_gen_config = get_pytorch_data_generation_config(n_iter=500,  # Number of iterations
                                                     data_gen_batch_size=32, # Batch size for data generation
                                                     image_padding=32, # image manipulation when generating data                                                     
                                                     # ... (other configuration parameters)
                                                     )

# Call the data generation function to generate images
generated_images = pytorch_data_generation_experimental(model=my_model,  # PyTorch model to generate data for
                                                        n_images=1024,  # Number of images to generate
                                                        output_image_size=224,  # Size of the output images
                                                        data_generation_config=data_gen_config  # Configuration for data generation
                                                        )
```

## Configuration Parameters

The `get_pytorch_data_generation_config()` function allows you to customize various configuration parameters for data generation. Here are the essential parameters that can be tailored to your specific needs:
- *n_iter*: Number of iterations for the data generation process.
- *optimizer*: The optimizer to use for the data generation process.
- *data_gen_batch_size*: Batch size for data generation.
- *initial_lr*: Initial learning rate for the optimizer.
- *output_loss_multiplier*: Multiplier for the output loss during optimization.
- *scheduler_type*: The type of scheduler to use.
- *bn_alignment_loss_type*: The type of BatchNorm alignment loss to use.
- *output_loss_type*: The type of output loss to use.
- *data_init_type*: The type of data initialization to use.
- *layer_weighting_type*: The type of layer weighting to use.
- *image_granularity*: The granularity of the images for optimization.
- *image_pipeline_type*: The type of image pipeline to use.
- *image_normalization_type*: The type of image normalization to use.
- *image_padding*: Padding to be applied to the images during data generation.
- *bn_layer_types*: List of BatchNorm layer types to be considered for data generation.
- *clip_images*: Whether to clip images during optimization.
- *reflection*: Whether to use reflection during optimization.

## Results Using Generated Data
### Experamental setup
##### Quantization Algorithms
The evaluation of the generated data involved using four quantization algorithms:

- **Regular Post Training Quantization (PTQ)** with 8 bit weights and activations.
- **Mixed Presicion** (MP) with a compression factor of x8
- **PTQ using a hardware-friendly Look-Up Table** for 4 bit weights, 8 bit activations.
- **Enhanced Post Training Quantization (EPTQ)** using 4 bit weights, 8 bit activations.

All setups where tested with symmetric weights and uniform activations.

To ensure reliable results, all experiments were averaged over 5 different random seeds (0-4). 

##### Data Generations Parameters
The evaluation was performed on the following neural network models:

- Resnet18 and Mobilenet v2 from the torchvision library.
- Yolo-n from ultralitics.
 
The default data generation configuration was used with 500 iterations for data generation (better results may be achieved with a larger iteration budget).
- For Resnet18 and Mobilenet v2, 1024 images were generated using a data generation batch size of 32 or 128 and a resolution of 224x224. 
- For Yolo-n, due to memory limitations, a batch size of 4 was used, and only 128 images were generated with a resolution of 640x640.

Please note that the choice of quantization algorithms and data generation parameters can have a significant impact on the results. The experimental setup provides a foundation for comparing the performance of different models and quantization techniques using the generated data.

|                Model                 |  Resnet18 (69.86)  |Resnet18 (69.86)     | Resnet18 (69.86)    | Resnet18 (69.86) |  Mobilenet v2 (71.89)  | Mobilenet v2 (71.89)  | Mobilenet v2 (71.89) | Mobilenet v2 (71.89) |  Yolo-v8-n (37.26)  |  Yolo-v8-n (37.26)   | Yolo-v8-n (37.26) |
|:------------------------------------:|:------------------:|:-------------------:|:-------------------:|:----------------:|:----------------------:|:---------------------:|:--------------------:|:--------------------:|:-------------------:|:--------------------:|:-----------------:|
|  Data type \ Quantization algorithm  |      PTQ W8A8      | MP compression x 8  |    PTQ LUT W4A8     |    EPTQ W4A8     |        PTQ W8A8        | MP compression x 8    |  PTQ LUT W4A8        |      EPTQ W4A8       |      PTQ W8A8       |     PTQ LUT W4A8     |     EPTQ W4A8     |
|              Real Data               |       69.49        |        58.48        |        66.24        |      69.30       |         71.168         |         64.52         |         64.4         |         70.6         |        36.23        |        29.79         |       25.82       |                  
|             Random Noise             |        8.3         |        43.58        |        12.6         |      11.13       |          7.9           |         30.02         |         7.14         |        11.30         |        27.45        |         4.15         |       2.68        |                  
|           Image Generation           |       69.51        |        58.57        |        65.70        |     69.07	    |         70.155         |         62.82         |        62.49         |        69.59         |        35.12        |        27.77         |       25.02       |                  
