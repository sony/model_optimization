# Quantization Troubleshooting for MCT

**Utilizing MCT for Model Compression:**

The Model Compression Toolkit (MCT) offers numerous functionalities to compress neural networks with minimal accuracy lost.
However, in some cases, the compressed model may experience a significant decrease in accuracy.
Fear not, as this lost accuracy can often be reclaimed by adjusting the quantization configuration or setup.

**Restoring Accuracy - Troubleshooting Steps:**

Outlined below are a series of steps aimed at recovering lost accuracy resulting from compression with MCT.
Some steps may be applicable to your model, while others may not.

*Note:*
Throughout this document we refer the user to notebooks using the Keras framework. There are similar notebooks for PyTorch.
All notebooks are available [here](https://github.com/sony/model_optimization/tree/main/tutorials/notebooks).

**Table of Contents:**
* [Representative Dataset](#representative-dataset)
* [Quantization Process](#quantization-process)
* [Model Structure Quantization Issues](#model-structure-quantization-issues)
* [Advanced Quantization Methods](#advanced-quantization-methods)
* [Debugging Tools](#debugging-tools)


___
## Representative Dataset
The representative dataset is used by the MCT to derive the threshold values of activation tensors in the model.

### 1. Representative dataset size & diversity:
If the representative dataset is too small, the thresholds will overfit and the accuracy on the evaluation dataset will degrade.
A similar overfitting may occur when the representative dataset isn't diverse enough (e.g. images from a single class, in a classification model).
In this case, the distribution of the target dataset and the representative dataset will not match, which might cause an accuracy degradation.

**Solution:**
Increase the number of samples in the representative dataset, or make sure that the samples are more diverse (e.g. include samples from all the classes, in a classification model).

### 2. Representative dataset mismatch:
Usually, the representative dataset is taken from the training set, and uses the same preprocess as the evaluation set.
If that's not the case, accuracy degradation is expected.

**Solution:**
Ensure that the preprocessing of the representative dataset is identical to the validation dataset, and its images are taken from the same domain.

___
## Quantization Process
There are many buttons to push and knobs to tune in MCT in case the default values fail to deliver.
The following are the usual suspects.

### 1. Outlier Removal

Outlier removal can become essential when quantizing activations,
particularly in scenarios where certain layers produce output activation tensors with skewed value distributions.
Such outliers can mess up the selection of quantization parameters and result in low accuracy in the quantized model.

In these scenarios, manually limit the activation thresholds using the `z_threshold` attribute in the `QuantizationConfig` in `CoreConfig`.

**Solution:**
Set a value to the `z_threshold`. Typical value range is between 5.0 and 20.0:
```python
core_config = mct.core.CoreConfig(mct.core.QuantizationConfig(z_threshold=8.0))
quantizted_model, quantization_info = mct.ptq.keras_post_training_quantization(model, representative_dataset,
                                                                               core_config=core_config)
```

### 2. Shift Negative Activation
Some activation functions are harder to quantize than others (e.g. the **swish**, **Leaky ReLU**, etc.).
The output of these activation layers contain negative values that only use a small part of the dynamic quantization range.
In these cases, shifting negative values to positive values can improve quantization error (saving an extra 1bit).

**Solution:** This exactly what the `shift_negative_activation_correction` flag does.
```python
core_config = mct.core.CoreConfig(mct.core.QuantizationConfig(shift_negative_activation_correction=True))
quantizted_model, quantization_info = mct.ptq.keras_post_training_quantization(model, representative_dataset,
                                                                               core_config=core_config)
```
*Note:* after activating this flag, you have a few more tweaks to its operation that you can control with the
`shift_negative_ratio`, `shift_negative_threshold_recalculation` & `shift_negative_params_searc` flags.
Read all about them in the [quantization configuration](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/common/quantization/quantization_config.py#L83) class description. 


### 3. Bias Correction
MCT applies bias correction by default to overcome induced bias shift caused by weights quantization.
The applied correction is an estimation of the bias shift that is computed based on (among other things) the collected statistical data generated with the representative dataset.
Therefore, the effect of the bias correction is sensitive to the distribution and size of the provided representative dataset.

**Solution**: Verify that the bias correction causes a degradation in accuracy by disabling it:
```python
core_config = mct.core.CoreConfig(mct.core.QuantizationConfig(weights_bias_correction=False))
quantizted_model, quantization_info = mct.ptq.keras_post_training_quantization(model, representative_dataset,
                                                                               core_config=core_config)
```
If you can increase your representative dataset and its distribution, it may restore accuracy.
If you don't have an option to increase or diversify your representative dataset, disabling the bias correction is recommended.

### 4. Threshold selection error method
The quantization threshold, which determines how data gets quantized, involves an optimization process driven by predefined objective metrics.
MCT defaults to employing the Mean-Squared Error (MSE) metric for threshold optimization,
however, it offers a range of alternative error metrics to accommodate different network requirements.

This flexibility becomes particularly crucial for activation quantization, where threshold selection spans the entire tensor and relies
on statistical insights for optimization. We advise you to consider other error metrics if your model is suffering from significant
accuracy degradation, especially if it contains unorthodox activation layers.

**Solution:**
For example, set "no clipping" error method to activations:
```python
quant_config = mct.core.QuantizationConfig(activation_error_method=mct.core.QuantizationErrorMethod.NOCLIPPING)
core_config = mct.core.CoreConfig(quantization_config=quant_config)
quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(..., core_config=core_config)
```
*Note:*
Some error methods (specifically, the KL-Divergence method) may suffer from extended runtime periods.
Opting for a different error metric could enhance threshold selection for one layer while potentially compromising another.
Therefore, thorough investigation and consideration are necessary.

___
## Model Structure Quantization Issues

### 1. Unbalanced "concatenation"
The concatenation operation can be a sensitive element in quantized models.
This is because it involves combining multiple tensors that may have significantly different value ranges, leading to potential inaccuracies in quantization.

For example, the quantization of a tensor that contains values ranging in (-1, 1) together with a tensor that ranges in (-64, 64) is less accurate than the quantization of each tensor separately.

**Solution**:
Identify problematic "concatenation" layers using the "network editor" API or by examining the quantization parameters of the input layers to the "concatenation" layer in the quantized model.
If you discover a concatenation layer with inputs that have notably different threshold values, consider either removing this concatenation, replacing its application to a later point in the model,
or balancing its input value ranges by adding a scaling operation.


___
## Advanced Quantization Methods

In certain scenarios, even if we've been thorough in following the quantization process, the degradation in accuracy is still
significant in the quantized model or its size surpassing the limitations of our deployment target (e.g., the IMX500).

MCT offers advanced features for mitigating these accuracy degradations, such as Mixed Precision and GPTQ.

### Mixed Precision Quantization

In mixed precision quantization, MCT will assign a different bit width to each weight in the model, depending on the weight's layer sensitivity
and a resource constraint defined by the user, such as target model size.

Check out the [mixed precision tutorial](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/mct_features_notebooks/keras/example_keras_mobilenet_mixed_precision.ipynb)
for more information and an implementation example. Following are a few tips for improving the mixed precision quantization.

#### 1. Using more samples in mixed precision quantization

By default, MCT employs 32 samples from the provided representative dataset for the mixed precision search.
Leveraging a larger dataset could enhance results, particularly when dealing with datasets exhibiting high variance.

**Solution:** 
Increase the number of samples (e.g. to 64 samples):
```python
mp_config = mct.core.MixedPrecisionQuantizationConfig(num_of_images=64)
core_config = mct.core.CoreConfig(mixed_precision_config=mp_config)
quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(..., core_config=core_config)
```
Please refer to the relevant notebooks (Keras mixed precision notebook and PyTorch mixed precision notebook) for comprehensive guidance.

*Note:*
Expanding the sample size may lead to extended runtime during the mixed precision search process.

#### 2. Mixed precision with model output loss objective

In mixed precision optimization, the aim is to determine an optimal bit-width assignment for each quantized layer to maximize the accuracy of the resulting model.
Traditionally, the mixed precision search relies on optimizing a loss function to achieve a high correlation with the actual model loss.
However, in scenarios involving high compression rates and complex models, the default objective may prioritize reducing the precision of the last layer, potentially leading to compromised results.

To overcome this challenge, MCT offers an API to adjust the mixed precision objective method.
By emphasizing a loss function that places greater importance on enhancing the model's quantized output, users can mitigate the risk of detrimental precision reductions in the last layer.
This feature is particularly suggested after an initial mixed precision run reveals a tendency towards reducing the bit-width of the last layer.

**Solution:**

```python
from model_compression_toolkit.core.common.mixed_precision import MpDistanceWeighting

mp_config = mct.core.MixedPrecisionQuantizationConfig(distance_weighting_method=MpDistanceWeighting.LAST_LAYER)
core_config = mct.core.CoreConfig(mixed_precision_config=mp_config)
quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(..., core_config=core_config)
```

#### 3. Enabling Hessian-based mixed precision

MCT offers a Hessian-based scoring mechanism to assess the importance of layers during the mixed precision search.
This feature can notably enhance mixed precision outcomes for certain network architectures.

**Solution:**
```python
mp_config = mct.core.MixedPrecisionQuantizationConfig(use_hessian_based_scores=True)
core_config = mct.core.CoreConfig(mixed_precision_config=mp_config)
quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(..., core_config=core_config)
```

*Note:*
Computing Hessian scores can be computationally intensive, potentially leading to extended mixed precision search runtime.
Furthermore, these scoring methods may introduce unexpected noise into the mixed precision process, necessitating a deeper understanding of the underlying mechanisms and potential recalibration of program parameters.


#### 3. Handling _"The model cannot be quantized to meet the specified target resource utilization"_ error

In case you encountered an Exception stating that the model cannot meet the target resource utilization, 
that means you are trying to run mixed precision quantization to reduce the model's memory footprint (either sum of all 
weights memory, maximum activation tensor memory, total weights and activation memory or number of bit-operations).
This process is activated based on a provided target resource utilization data ([ResourceUtilization](./model_compression_toolkit/core/common/mixed_precision/resource_utilization_tools/resource_utilization.py)).
The error is stating that the provided target is too strict, and the model cannot be quantized, based on the provided [TPC](./model_compression_toolkit/target_platform_capabilities/README.md) and quantization configurations, to meet the desired restrictions.

**Solution:**
There are several steps that you can try to figure out what the problem is and fix it:
First of all, verify that you intended to run mixed precision, if not, you shouldn't provide a target resource utilization.
If you did attempt to compress the model to a specific target, then verify the resource utilization object that you passed to the MCT: 
1. Verify that it include a value only for the resource that you want to restrict.
2. Validate the actual compression ratio of the values that you provided.

It may be worth to try and soften the restrictions (increase the target values or remove restrictions on certain resources)
as an attempt to see if there is a more general problem or whether the problem is with the tightness of the restriction.

If all the above verifications checked out, you might want to look that the provided TPC for any inconsistencies.
For example, maybe you are trying to restrict the activation memory size, but there are layers that do not provide 
multiple configuration candidates for quantizing the activation via mixed precision. 

### GPTQ - Gradient-Based Post Training Quantization

When PTQ (either with or without mixed precision) fails to deliver the required accuracy, GPTQ is potentially the remedy.
In GPTQ, MCT will finetune the model's weights and quantization parameters for improved accuracy. The finetuning process
will only use the label-less representative dataset.

Check out the [GPTQ tutorial](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/mct_features_notebooks/keras/example_keras_mobilenet_gptq.ipynb) for more information and an implementation example.

*Note #1*: The finetuning process will take **much** longer to finish than PTQ. As in any finetuning, some hyperparameters optimization may be required.

*Note #2*: You can use mixed precision and GPTQ together.


___
## Debugging Tools

### 1. Network Editor

To pinpoint the possible cause of accuracy degradation in your model, it may be necessary to isolate the quantization settings of specific layers.
However, this cannot be accomplished using MCT's main API.

For instance, accuracy might suffer due to quantization of certain layer types, such as `Swish` for activations or `Conv2d` for weights.
To address this issue and manipulate individual layers within the network, you can utilize the "Network Editor" API.

**Implementation**:

Using the network editor API (mct.core.network_editor) you can define a set of "rules" to apply on the network, using provided filters.

Please refer to our [tutorial](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/mct_features_notebooks/keras/example_keras_network_editor.ipynb) for more details.

A use-case example to understand if a certain layer causes accuracy degradation - set the output quantization bits of the layer's type to 16 bits (instead of the default 8 bits defined in the TPC).
If the accuracy improves, then it is pointing to that layer for causing the issue.

*Note:* This is mainly a debugging tool and should not be used as an API for quantizing a model for deployment (may result in an unstable behavior).

### 2. Debug mixed precision quantization
MCT offers an API to provide a pre-configured mixed precision bit-width assignment for quantizing the model.
This capability can be invaluable for identifying limitations in model quantization by setting specific layers to low bit-widths and examining the resulting model accuracy.

**Solution:**
To override the bit-width configuration, utilize the configuration_overwrite argument within the MixedPrecisionQuantizationConfig object.
Ensure that the provided configuration follows a specific format: a list of integers representing the desired bit-width candidate index for each layer.

For example, in a model with 3 layers offering multiple bit-width candidates for mixed precision quantization (8, 4, and 2),
a configuration of [0, 1, 2] would quantize the first layer to 8 bits (the highest bit-width option), the second layer to 4 bits, and the third layer to 2 bits.
```python
mp_config = mct.core.MixedPrecisionQuantizationConfig(configuration_overwrite=[0, 1, 2])
core_config = mct.core.CoreConfig(mixed_precision_config=mp_config)
quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(..., core_config=core_config)
```

*Note:* This feature necessitates a proficient understanding of mixed precision execution. Users must ensure that:
* The provided configuration adheres to the correct list format.
* A target KPI is provided, to activate a mixed precision search.
