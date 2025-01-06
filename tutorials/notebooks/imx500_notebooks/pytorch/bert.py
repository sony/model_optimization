from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, default_data_collator, \
    DataCollatorForSeq2Seq
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset, DataLoader, Dataset
from torch.utils.data import Subset
from datasets import load_dataset
import random
import model_compression_toolkit as mct
from typing import Iterator, Tuple, List
from time import time
import torch
# import sys
# sys.setrecursionlimit(3000)


def get_glue_dataset_num_labels():
    if is_regression_task:
        return 1
    return len(plain_dataset["train"].features["label"].names)

quantization_dataset_name = "wikitext"
batch_size = 1
dataset_name = "glue"
task_name = "mrpc"
model_name = 'bert-base-uncased'
# model_name = '/data/projects/swat/network_database/Pytorch/internal/language_models/GLUE_benchmark/BERT_base/mrpc/checkpoint-345'
is_regression_task = task_name == "stsb"
plain_dataset = load_dataset(dataset_name, task_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name,
                                                           num_labels=get_glue_dataset_num_labels(), torchscript=True)
embedding = model.bert.embeddings
model.eval()

#####################################tokenize dataset######################################################

def get_wikitext_tokenized_dataset():

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    tokenized_datasets = testdata.map(tokenize_function, batched=True)
    train_dataloader = DataLoader(tokenized_datasets, shuffle=False, batch_size=batch_size, collate_fn=data_collator)
    return train_dataloader


def get_glue_tokenized_dataset():
    def get_representative_dataloader(tokenized_dataset, num_samples, shuffle):
        if num_samples:
            tokenized_dataset = Subset(tokenized_dataset,
                                       list(random.sample(range(len(tokenized_dataset)), num_samples)))

        dataset_loader = DataLoader(
            tokenized_dataset,
            shuffle=shuffle,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=4,
        )
        return dataset_loader
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
    train_tokenized_dataset, eval_tokenized_dataset = _tokenize_glue_dataset(dataset, task_name, tokenizer)
    representative_dataloader: DataLoader = get_representative_dataloader(train_tokenized_dataset, num_samples=20, shuffle=False)
    return representative_dataloader

data_collator = default_data_collator
if quantization_dataset_name == "wikitext":
    representative_dataloader = get_wikitext_tokenized_dataset()
else:
    representative_dataloader = get_glue_tokenized_dataset()
#####################################get representative dataset######################################################


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
                input_data = next(ds_iter)
                input_embeddings = embedding(input_data["input_ids"])
                yield input_embeddings
                # yield next(ds_iter)
            except StopIteration:
                return

    return representative_dataset

representative_dataset: Iterator = get_representative_dataset(2, representative_dataloader)

#####################################quantize model######################################################
start = time()
# Perform post training quantization with the default configuration
quant_model, _ = mct.ptq.pytorch_post_training_quantization(model, representative_dataset)
print('Quantized model is ready')

#####################################export model######################################################


mct.exporter.pytorch_export_model(model=quant_model,
                                  save_model_path='./bert_qmodel_wo_embeds_2mamtul_subs.onnx',
                                  repr_dataset=representative_dataset,
                                  onnx_opset_version=20)
end = time()
runtime_seconds = end - start
print(f"Finished model quantization! Total runtime: {(runtime_seconds):.2f} seconds ({runtime_seconds / 60:.2f} minutes)")
"""
bert-mct issues map:
attention substitution fails
node with slice(0, add, None) in op_call_args
nodes with output_shape=[]
output with single value: [128] and not a tensor
WARNING:Model Compression Toolkit:Skipping bias correction due to valiation problem.
recursion error on run, but not on debug mode
"""
