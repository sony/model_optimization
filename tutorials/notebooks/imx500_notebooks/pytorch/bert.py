from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, default_data_collator
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from datasets import load_dataset
import random
import model_compression_toolkit as mct
from typing import Iterator, Tuple, List
import torch

def get_glue_dataset_num_labels():
    if is_regression_task:
        return 1
    return len(plain_dataset["train"].features["label"].names)

batch_size = 4
dataset_name = "glue"
task_name = "mrpc"
model_name = 'bert-base-uncased'
is_regression_task = task_name == "stsb"
plain_dataset = load_dataset(dataset_name, task_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name,
                                                           num_labels=get_glue_dataset_num_labels(), torchscript=True)

model.eval()

#####################################tokenize dataset######################################################

def _get_glue_task_to_keys():
    return {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        # "wnli": ("sentence1", "sentence2"),
    }

def _tokenize_glue_dataset(dataset, task_name, tokenizer):

    task_to_keys = _get_glue_task_to_keys()
    sentence1_key = task_to_keys[task_name][0]
    sentence2_key = task_to_keys[task_name][1]

    def _tokenize(batch):
        args = ((batch[sentence1_key],) if sentence2_key is None else (batch[sentence1_key], batch[sentence2_key]))
        tokenized_batch = tokenizer(*args, padding="max_length", max_length=128, truncation=True)
        return tokenized_batch

    tokenized_dataset = dataset.map(
        _tokenize,
        batched=True,
        desc="Running tokenizer on dataset",
    )
    train_tokenized_dataset = tokenized_dataset["train"]
    eval_tokenized_dataset = tokenized_dataset["validation_matched" if task_name == "mnli" else "validation"]
    return train_tokenized_dataset, eval_tokenized_dataset


dataset = load_dataset(dataset_name, task_name, trust_remote_code=True)
train_tokenized_dataset, eval_tokenized_dataset = _tokenize_glue_dataset(plain_dataset, task_name, tokenizer)
data_collator = default_data_collator

#####################################get representative dataset######################################################


def get_representative_dataloader(tokenized_dataset, num_samples, shuffle):
    if num_samples:
        tokenized_dataset = Subset(tokenized_dataset, list(random.sample(range(len(tokenized_dataset)), num_samples)))

    dataset_loader = DataLoader(
        tokenized_dataset,
        shuffle=shuffle,
        collate_fn=data_collator,
        batch_size=batch_size,
        num_workers=4,
    )
    return dataset_loader

def get_representative_dataset(n_iter: int, dataset_loader: Iterator[Tuple]):
    """
    This function creates a representative dataset generator. The generator yields numpy
        arrays of batches of shape: [Batch, H, W ,C].
    Args:
        n_iter: number of iterations for MCT to calibrate on
    Returns:
        A representative dataset generator
    """

    def representative_dataset() -> Iterator[List]:
        ds_iter = iter(dataset_loader)
        for _ in range(n_iter):
            try:
                yield next(ds_iter)
            except StopIteration:
                return

    return representative_dataset
representative_dataloader: DataLoader = get_representative_dataloader(train_tokenized_dataset, num_samples=20, shuffle=False)
representative_dataset: Iterator = get_representative_dataset(8, representative_dataloader)

#####################################quantize model######################################################

# Perform post training quantization with the default configuration
quant_model, _ = mct.ptq.pytorch_post_training_quantization(model, representative_dataset)
print('Quantized model is ready')

#####################################export model######################################################


mct.exporter.pytorch_export_model(model=quant_model,
                                  save_model_path='./bert_qmodel.onnx',
                                  repr_dataset=representative_dataset,
                                  onnx_opset_version=20)

"""
bert-mct issues map:
attention substitution fails
node with slice(0, add, None) in ol_call_args
nodes with output_shape=[]
output with single value: [128] and not a tensor
WARNING:Model Compression Toolkit:Skipping bias correction due to valiation problem.
recursion error on run, but not on debug mode
"""