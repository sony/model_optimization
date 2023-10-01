# Gradient-based Post-Training Quantization (GPTQ)

The GPTQ package allows to run advanced quantization algorithm, in which the quantized parameters rounding is optimized after the initial quantization is applied.

GPTQ preserves the principals of post-training quantization - it uses a small. unlabeled dataset, and does not require any additional information regarding the model.

MCT provides several symmetric and uniform [trainable quantizers](../trainable_infrastructure/README.md) that allow to retrain the model. 

## Enhanced Post-Training Quantization (EPTQ)

EPTQ is an in-house developed gradient-based quantization scheme that allows to retrain the quantized parameters [1].
It is implemented for both Keras and PyTorch frameworks, and can be utilized via the GPTQ API.
EPTQ is the default algorithm used when running GPTQ.

EPTQ utilizes Hessian information (Hessian matrix trace w.r.t. pre-defined layer's activation tensors) to construct an optimization objective that gives special attention to the layers that are more sensitive to quantization.
The Hessian information is approximated in a label-free manner, thus, does not require any additional data beside the unlabeled data provided to the original PTQ method. 

<img src="../../docsrc/images/eptq_overview.svg" width="10000">


## GPTQ Usage

For an example of how to use GPTQ in MCT with TensorFlow or PyTorch on various models and tasks, check out the [tutorials package](../../tutorials),
where you can find extended explanation and notebook examples for running GPTQ, along with a [quick-start](../../tutorials/quick_start/README.md) framework for simple execution.

## References

[1] Gordon, O., Habi, H. V., & Netzer, A., 2023. [EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian. arXiv preprint](https://arxiv.org/abs/2309.11531)