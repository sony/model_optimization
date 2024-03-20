# Quantization Troubleshooting for MCT

**Compressing a model with MCT:**
The Model Compression Toolkit (MCT) is a great tool for compressing your model with only a few lines of code, with a minimal impact on accuracy.
However, sometimes the compressed model's accuracy decrease is too large. But don't despair, the lost accuracy may be recovered by adjusting the quantization configuration
or quantization setup.

**Recover Accuracy: Troubleshooting:**
The following list contains a set of operations that may recover lost accuracy do to compression with MCT.
Some operations might be relevant to your model and others may not.

## PTQ Features

### Representative Dataset
The representative dataset is used by the MCT to derive the threshold values of activation tensors in the model.

#### Representative dataset size & diversity:
If the representative dataset is too small, the thresholds will overfit and the accuracy will on the evaluation dataset will degrade.
A similar overfitting may occur when the representative dataset isn't diverse enough (e.g. images from a single class, in a classification model).
In this case, the distribution of the target dataset and the representative dataset will not match and accuracy will degrade.

**Solution:**
Increase the number of inputs in the representative dataset, or make sure the inputs are more diverse (e.g. include all the classes, in  a classification model).

#### Representative dataset mismatch:
Usually, the representative dataset is taken from the training set, and uses the same preprocess as the evaluation set.
If that's not the case, accuracy degradation is expected.

**Solution:**
Make sure the representative dataset preprocess is the same, and its images are taken from the same domain.


### Quantization Process
There are many buttons to push and knobs to tune in MCT in case the default values fail to deliver.
The following are the usual suspects.

#### Outlier Removal
In some models, you may need to manually limit the activation thresholds using the `z_threshold` attribute in the `QuantizationConfig` in `CoreConfig`.

**Solution:**
Set a value to the `z_threshold`. Typical value range is between 5.0 and 40.0:
```python
core_config = mct.core.CoreConfig(mct.core.QuantizationConfig(z_threshold=17.0))
quantizted_model, quantization_info = mct.ptq.keras_post_training_quantization(model, representative_dataset,
                                                                               core_config=core_config)
```

#### Shift Negative Activation
Some activation functions are harder to quantize than others (e.g. the **swish**, **Leaky ReLU**, etc.).
These activation layers have negative values that use only a small part of the dynamic range.
In these cases, shifting negative values to positive values can improve quantization error (saving an extra 1bit).

**Solution:** This exactly what the `shift_negative_activation_correction` flag does.
```python
core_config = mct.core.CoreConfig(mct.core.QuantizationConfig(shift_negative_activation_correction=True))
quantizted_model, quantization_info = mct.ptq.keras_post_training_quantization(model, representative_dataset,
                                                                               core_config=core_config)
```
*note:* after activating this flag, you have a few more tweaks to its operation that you can control with the
`shift_negative_ratio`, `shift_negative_threshold_recalculation` & `shift_negative_params_searc` flags.
Read all about them in the [quantization config](https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/common/quantization/quantization_config.py#L83) 

Additional care should be taken when using this feature.????

### Bias Correction
MCT applies bias correction by default to overcome induced bias shift caused by weights quantization.
The applied correction is an estimation of the bias shift that is computed based on (among other things) the collected statistical data generated with the representative dataset.
Therefore, the effect of the bias correction is sensitive to the distribution and size of the provided representative dataset.

**Solution**: First, verify the bias correction causes a degradation in accuracy by disabling it:
```python
core_config = mct.core.CoreConfig(mct.core.QuantizationConfig(weights_bias_correction=False))
quantizted_model, quantization_info = mct.ptq.keras_post_training_quantization(model, representative_dataset,
                                                                               core_config=core_config)
```
If you can increase your representative dataset and its distribution, it may restore accuracy.
If you don't have an option to increase or diversify your representative dataset, disable the bias correction.


## Common Issues

### Unbalanced "concatenation"
The concatenation operation can be a sensitive element in quantized models.
This is because it involves combining multiple tensors that may have significantly different value ranges, leading to potential inaccuracies in quantization.

For example, the quantization of a tensor that contains values ranging in (-1, 1) together with a tensor that ranges in (-64, 64) is less accurate than the quantization of each tensor separately.

**Solution**:
You can identify problematic "concatenation" layers using the "network editor" API or by examining the quantization parameters of the input layers to the "concatenation" layer in the quantized model.
If you discover a concatenation layer with inputs that have notably different threshold values, consider either removing this concatenation, replacing its application to a later point in the model, or balancing its input value ranges by additional an scaling operation.


### Pytorch model - common torch.FX errors
When implementing quantization on a PyTorch model, MCT's initial step involves converting the model into a graph representation using `torch.fx`.
However, `torch.fx` comes with certain common limitations, with the primary one being its requirement for the computational graph to remain static.

Despite these limitations, some adjustments can be made to facilitate MCT quantization.

**Solution**: (assuming you have access to the model's code)

Check the `torch.fx` error, and search for an identical replacement:

An `if` statement in a module's `forward` method might can be easily skipped.
The `list()` Python method can be replaced with a concatenation operation [A, B, C].


## Debugging Tools

### Network Editor

To pinpoint the possible cause of accuracy degradation in your model, it may be necessary to isolate the quantization settings of specific layers.
However, this cannot be accomplished using MCT's main API.

For instance, accuracy might suffer due to quantization of certain layer types, such as `Swish` for activations or `Conv2d` for weights.
To address this issue and manipulate individual layers within the network, you can utilize the "Network Editor" API.

**Implementation**:

Using the network editor API (mct.core.network_editor) you can define a set of "rules" to apply on the network, using provided filters.

Please refer to our [tutorial](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/keras/debug_tools/example_keras_network_editor.ipynb) for more details.

A use-case example to understand if a certain layer causes accuracy degradation - set the output quantization bits of the layer's type to 16 bits (instead of the default 8 bits defined in the TPC).
If the accuracy improves, then it is pointing to that layer for causing the issue.

*Note:* This is mainly a debugging tool and should not be used as an API for quantizing a model for deployment (may result in an unstable behavior).


## Advanced Quantization Methods

Sometimes, we did everything right in the quantization process, but either the quantized model accuracy degradation is too high
or the quantized model size is too large for our deployment target (e.g. the IMX500).
MCT offers advanced features for handing these issues, such as mixed-precesion and GPTQ.

### Mixed Precision Quantization

In mixed precision quantization, MCT will assign a different number of bit widths to each weight in the model, depending on the weight's layer sensitivity
and a constraint defined by the user, such as target model size.

Check out the [mixed precision tutorial](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/keras/ptq/example_keras_mobilenet_mixed_precision.ipynb)
for more information and an implementation example.

### GPTQ - Gradient-Based Post Training Quantization

When PTQ (either with or without mixed-precision) fails to deliver the required accuracy, GPTQ is potentially the remedy.
In GPTQ, MCT will finetune the model's weights and quantization parameters for improved accuracy. The finetuning process
will only use the label-less representative dataset.

Check out the [GPTQ tutorial](https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/keras/gptq/example_keras_mobilenet_gptq.ipynb) for more information and an implementation example.

*Note #1*: The finetuning process will take longer to finish than PTQ. As in any finetuning, some hyperparameters optimization may be required.

*Note #2*: You can use mixed-precision and GPTQ together.


**TODO:** Add Exporter stuff 