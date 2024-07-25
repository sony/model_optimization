# Data Generation Library for Model Compression Toolkit (MCT)

The Data Generation Library for the Model Compression Toolkit (MCT) is a powerful Python package designed to generate synthetic data. This library supports both PyTorch and Keras. 

The effectiveness of Post-Training Quantization (PTQ) depends on the availability of high-quality representative data. However, practical scenarios often limit access to such data due to privacy or security concerns. Zero-Shot Quantization (ZSQ) provides an alternative by enabling PTQ with synthetic data. We leverage the statistics stored in batch normalization (BN) layers within pre-trained models to generate synthetic data that faithfully represents the model's training data characteristics.

## Key Features

- **Synthetic Data Generation**: Generate synthetic data based on the statistics from the model's BN layers.
- **Complementary to Compression Tasks**: Use the generated data to improve various compression tasks, including quantization and pruning of neural networks.
- **User-Friendly Interface**: The library provides easy-to-use functions, making data generation seamless and accessible to researchers, developers, and engineers.
- **Integrates with MCT**: The Data Generation Library seamlessly integrates with the Model Compression Toolkit, enhancing its capabilities for neural network optimization and deployment.

## Installation
To install the package, you can use pip and install the latest stable release of the Model Compression Toolkit:
```bash
pip install model-compression-toolkit
```

## Usage
### PyTorch
Explore a Jupyter Notebook example showcasing data generation with ResNet18, including visualizations, and a practical example of Post Training Quantization:
* [Data Generation for Resnet18 with PyTorch](../../tutorials/notebooks/mct_features_notebooks/pytorch/example_pytorch_data_generation.ipynb).

Below, you will find a straightforward usage example:
```python
import model_compression_toolkit as mct

# Set the configuration parameters for data generation
data_gen_config = mct.data_generation.get_pytorch_data_generation_config(
    n_iter=500,                               # Number of iterations
    data_gen_batch_size=32,                   # Batch size for data generation
    extra_pixels=32,                          # image manipulation when generating data                                                     
    # ... (other configuration parameters)
)

# Call the data generation function to generate images
generated_images = mct.data_generation.pytorch_data_generation_experimental(
    model=my_model,                          # PyTorch model to generate data for
    n_images=1024,                           # Number of images to generate
    output_image_size=224,                   # Size of the output images
    data_generation_config=data_gen_config   # Configuration for data generation
)
```

### Keras

```python
import model_compression_toolkit as mct

# Set the configuration parameters for data generation
data_gen_config = mct.data_generation.get_keras_data_generation_config(
    n_iter=500,  # Number of iterations
    data_gen_batch_size=32,  # Batch size for data generation
    extra_pixels=32,  # image manipulation when generating data                                                     
    # ... (other configuration parameters)
)

# Call the data generation function to generate images
generated_images = mct.data_generation.keras_data_generation_experimental(
    model=my_model,  # PyTorch model to generate data for
    n_images=1024,  # Number of images to generate
    output_image_size=224,  # Size of the output images
    data_generation_config=data_gen_config  # Configuration for data generation
)
```

## Configuration Parameters

The `get_pytorch_data_generation_config()` and `get_keras_data_generation_config()` functions allow you to customize various configuration parameters for data generation. Here are the essential parameters that can be tailored to your specific needs:
- **'n_iter'** (int):  The number of iterations for the data generation optimization process. Controls the number of iterations used during the optimization process for generating data. Higher values may improve data quality at the cost of increased computation time.
- **'optimizer'** (Optimizer): The optimizer used during data generation to update the generated images. Specifies the optimization algorithm used to update the generated images during the data generation process. Common optimizers include RAdam, Adam, SGD, etc.
- **'data_gen_batch_size'** (int): The batch size used during data generation optimization. Determines the number of images processed in each optimization step. A larger batch size may speed up the optimization but requires more memory.
- **'initial_lr'** (float): The initial learning rate used by the optimizer during data generation. Controls the initial step size for updating the generated images during optimization. The learning rate affects the speed and stability of the optimization process.
- **'output_loss_multiplier'** (float): A multiplier used to scale the output loss term in the overall loss function. Adjusts the relative importance of the output loss component in the optimization process. 
- **'scheduler_type'**(SchedularType): The type of learning rate scheduler used during data generation. Determines the strategy for adjusting the learning rate during optimization. 
- **'bn_alignment_loss_type'**(BatchNormAlignemntLossType): The type of BatchNorm alignment loss used in the optimization process. Specifies the method for calculating the alignment loss between the original and generated BatchNorm statistics.
- **'output_loss_type'** (OutputLossType): The type of output loss to use.
- **'data_init_type'** (DataInitType):  The type of image initialization used to initialize the generated data. Specifies the method for generating the initial set of images used for data generation. Options include random noise, diverse set, etc.
- **'layer_weighting_type'** (LayerWeightingType): The type of layer weighting used in the optimization process. Determines how the layers of the model are weighted during the loss calculation. Common options include average weighting and gradient based weighting.
- **'image_granularity'** (ImageGranularity): The granularity at which data generation is performed. Specifies whether data generation is performed per image or for all images together. This affects the statistics calculations and optimization process.
- **'image_pipeline_type'** (ImagePipelineType): The type of image processing pipeline used during data generation. Specifies the pipeline used to process the images before feeding them into the model during data generation. Options include crop and flip, resize, etc.
- **'image_normalization_type'** (ImageNormalizationType): The type of image normalization applied to the images during data generation. Specifies the method used to normalize the pixel values of the images. Common options include torchvision normalization, custom mean, and standard deviation normalization, etc.
- **'extra_pixels'** (int):  The number of extra pixels added to the input image size during data generation.
- **'bn_layer_types'** (List): List of BatchNorm layer types present in the model. Specifies the types of BatchNorm layers in the model that require alignment between original and generated statistics.
- **'clip_images'** (bool):  Indicates whether the generated images should be clipped to a valid grid of pixel values. Controls whether the generated images are restricted to a valid range of pixel values. Clipping can improve image quality and avoid unrealistic pixel values.

## Results Using Generated Data
## PyTorch
### Experimental setup
##### Quantization Algorithms
Four quantization algorithms were utilized to evaluate the generated data:

- **Post Training Quantization (PTQ)** with 8 bit weights and activations.
- **Mixed Presicion (MP)**  with a total compression factor of x8.
- **Quantization using a hardware-friendly Look-Up Table (LUT)** for 4 bit weights, 8 bit activations.
- **Enhanced Post Training Quantization (EPTQ)** using 4 bit weights, 8 bit activations.

All setups were tested with symmetric weights and uniform activation quantizers.

To ensure reliable results, all experiments were averaged over 5 different random seeds (0-4). 

##### Data Generations Parameters
The evaluation was performed on the following neural network models:

- ResNet18, MobileNet V2, MobileNet V3, MNASNet and GoogleNet from the torchvision library.
- Yolo-n from [ultralytics](https://github.com/ultralytics/ultralytics).

The quantization algorithms were tested using real data and generated data.
The generated data was produced using the default data generation configuration with 1000 iterations (better results may be achieved with a larger iteration budget).
- For ResNet18, MobileNet V2, MobileNet V3, MNASNet and GoogleNet, 1024 images were generated using a data generation batch size of 128 and a resolution of 224x224. 
- For Yolo-n, due to memory limitations, a batch size of 4 was used, and only 128 images were generated with a resolution of 640x640.

Please note that the choice of quantization algorithms and data generation parameters can have a significant impact on the results. The experimental setup provides a foundation for comparing the performance of different models and quantization techniques using the generated data.

[//]: # (|                    Model &#40;float&#41;                    |  Resnet18 &#40;69.86&#41;  |Resnet18 &#40;69.86&#41;     | Resnet18 &#40;69.86&#41;    | Resnet18 &#40;69.86&#41; |  Mobilenet v2 &#40;71.89&#41;  | Mobilenet v2 &#40;71.89&#41;  | Mobilenet v2 &#40;71.89&#41; | Mobilenet v2 &#40;71.89&#41; |  Yolo-v8-n &#40;37.26&#41;  |  Yolo-v8-n &#40;37.26&#41;   | Yolo-v8-n &#40;37.26&#41; |)

[//]: # (|:---------------------------------------------------:|:------------------:|:-------------------:|:-------------------:|:----------------:|:----------------------:|:---------------------:|:--------------------:|:--------------------:|:-------------------:|:--------------------:|:-----------------:|)

[//]: # (| Data type &#40;rows&#41; \ Quantization algorithm &#40;columns&#41; |      PTQ W8A8      | MP compression x 8  |    PTQ LUT W4A8     |    GPTQ W4A8     |        PTQ W8A8        | MP compression x 8    |  PTQ LUT W4A8        |      GPTQ W4A8       |      PTQ W8A8       |     PTQ LUT W4A8     |     GPTQ W4A8     |)

[//]: # (|                      Real Data                      |       69.49        |        58.48        |        66.24        |      69.30       |         71.168         |         64.52         |         64.4         |         70.6         |        36.23        |        29.79         |       25.82       |                  )

[//]: # (|                    Random Noise                     |        8.3         |        43.58        |        12.6         |      11.13       |          7.9           |         30.02         |         7.14         |        11.30         |        27.45        |         4.15         |       2.68        |                  )

[//]: # (|                  Image Generation                   |       69.51        |        58.57        |        65.70        |     69.07	    |         70.155         |         62.82         |        62.49         |        69.59         |        35.12        |        27.77         |       25.02       |                  )

<table>
    <thead>
        <tr>
            <th rowspan="2" style="text-align:center;">Model (Float)</th>
            <th rowspan="2" style="text-align:center;">Data Generation Time (Minutes)</th>
            <th colspan="2" style="text-align:center;">PTQ W8A8</th>
            <th colspan="2" style="text-align:center;">EPTQ W4A8</th>
            <th colspan="2" style="text-align:center;">Mixed Precision Compression x8</th>
            <th colspan="2" style="text-align:center;">Look Up Table W4A8</th>
        </tr>
        <tr>
            <th style="text-align:center;">Real Data</th>
            <th style="text-align:center;">Data Generation</th>
            <th style="text-align:center;">Real Data</th>
            <th style="text-align:center;">Data Generation</th>
            <th style="text-align:center;">Real Data</th>
            <th style="text-align:center;">Data Generation</th>
            <th style="text-align:center;">Real Data</th>
            <th style="text-align:center;">Data Generation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align:center;">ResNet18 (69.86)</td>
            <td style="text-align:center;">12.5</td>
            <td style="text-align:center;">69.6</td>
            <td style="text-align:center;">69.57</td>
            <td style="text-align:center;">68.906</td>
            <td style="text-align:center;">68.64</td>
            <td style="text-align:center;">62.464</td>
            <td style="text-align:center;">61.834</td>
            <td style="text-align:center;">66.209</td>
            <td style="text-align:center;">64.739</td>
        </tr>
        <tr>
            <td style="text-align:center;">MobileNetV2 (71.89)</td>
            <td style="text-align:center;">26.5</td>
            <td style="text-align:center;">71.142</td>
            <td style="text-align:center;">71.388</td>
            <td style="text-align:center;">69.333</td>
            <td style="text-align:center;">68.599</td>
            <td style="text-align:center;">64.319</td>
            <td style="text-align:center;">66.355</td>
            <td style="text-align:center;">63.963</td>
            <td style="text-align:center;">64.136</td>
        </tr>
        <tr>
            <td style="text-align:center;">MobileNet V3 (74.056)</td>
            <td style="text-align:center;">22</td>
            <td style="text-align:center;">73.592</td>
            <td style="text-align:center;">73.628</td>
            <td style="text-align:center;">71.644</td>
            <td style="text-align:center;">70.581</td>
            <td style="text-align:center;">66.748</td>
            <td style="text-align:center;">65.358</td>
            <td style="text-align:center;">63.364</td>
            <td style="text-align:center;">63.292</td>
        </tr>
        <tr>
            <td style="text-align:center;">MNASNet (73.412)</td>
            <td style="text-align:center;">27</td>
            <td style="text-align:center;">73.186</td>
            <td style="text-align:center;">69.377</td>
            <td style="text-align:center;">71.081</td>
            <td style="text-align:center;">69.076</td>
            <td style="text-align:center;">66.811</td>
            <td style="text-align:center;">67.332</td>
            <td style="text-align:center;">66.21</td>
            <td style="text-align:center;">65.862</td>
        </tr>
        <tr>
            <td style="text-align:center;">GoogleNet (69.758)</td>
            <td style="text-align:center;">27.5</td>
            <td style="text-align:center;">69.71</td>
            <td style="text-align:center;">69.332</td>
            <td style="text-align:center;">68.757</td>
            <td style="text-align:center;">68.286</td>
            <td style="text-align:center;">59.854</td>
            <td style="text-align:center;">54.828</td>
            <td style="text-align:center;">65.904</td>
            <td style="text-align:center;">65.862</td>
        </tr>
        <tr>
            <td style="text-align:center;">Yolo-v8-n (37.26)</td>
            <td style="text-align:center;">50</td>
            <td style="text-align:center;">36.23</td>
            <td style="text-align:center;">35.12</td>
            <td style="text-align:center;">25.82</td>
            <td style="text-align:center;">25.02</td>
            <td style="text-align:center;">-</td>
            <td style="text-align:center;">-</td>
            <td style="text-align:center;">29.79</td>
            <td style="text-align:center;">27.77</td>
        </tr>
    </tbody>
</table>



## Keras
### Experimental setup
##### Quantization Algorithms
Post Training Quantization (PTQ) algorithm was utilized to evaluate the generated data:

All experiments were tested with symmetric weights and Power-Of-Two activation quantizers.

To ensure reliable results, all experiments were averaged over 5 different random seeds (0-4). 

##### Data Generations Parameters
The evaluation was performed on the following neural network models:

- Mobilenet and Mobilenetv2 from the keras applications library.

The quantization algorithms were tested using three different data types as input: real data, random noise, and generated data.
The generated data was produced using the default data generation configuration with 500 iterations (better results may be achieved with a larger iteration budget).
- 1024 images were generated using a data generation batch size of 32 and a resolution of 224x224. 

|                    Model (float)                    | Mobilenet (70.558) | Mobilenetv2 (71.812) |
|:---------------------------------------------------:|:------------------:|:--------------------:|
| Data type (rows) \ Quantization algorithm (columns) |      PTQ W8A8      |       PTQ W8A8       |
|                      Real Data                      |       70.427       |        71.599        |                  
|                    Random Noise                     |       58.938       |        70.932        |                  
|                  Image Generation                   |       70.39        |        71.574        |                  
