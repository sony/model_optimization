# Gradient-based Post-Training Quantization (GPTQ)
GPTQ is a core feature of MCT, offering an advanced quantization algorithm that optimizes parameter rounding after the initial quantization process. It preserves the principles of post-training quantization, utilizing a small, unlabeled dataset without the need for additional model information.

MCT provides a variety of symmetric [trainable quantizers](../trainable_infrastructure/README.md) that allow you to retrain your model efficiently.

## Enhanced Post-Training Quantization (EPTQ)

EPTQ is an in-house developed gradient-based quantization scheme that allows to retrain the quantized parameters [1].
It is implemented for both Keras and PyTorch frameworks, and can be utilized via the GPTQ API.
EPTQ is the default algorithm used when running GPTQ.

EPTQ leverages Hessian information (Hessian matrix trace with respect to pre-defined layer activation tensors) to construct an optimization objective that prioritizes layers more sensitive to quantization.
The Hessian information is approximated in a label-free manner, thus, does not require any additional data beside the unlabeled data provided to the original PTQ method. 

<img src="../../docsrc/images/eptq_overview.svg" width="10000">


## GPTQ Usage

For detailed examples and tutorials on using GPTQ in MCT with TensorFlow or PyTorch across various models and tasks, please refer to the [tutorials package](../../tutorials). You will find comprehensive explanations, notebook examples, and a [quick-start guide](../../tutorials/quick_start/README.md) for straightforward execution.

## References

[1] Gordon, O., Habi, H. V., & Netzer, A., 2023. [EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian. arXiv preprint](https://arxiv.org/abs/2309.11531)