# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import is_main_process


from transformers import BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
import torch

from .abstract_model import (
    Typer,
    SpanClassifierOutput,
    SpanClassifierDataTrainingArguments,
)


class BertForSpanInsideClassification(BertForSequenceClassification):
    def forward(
        self,
        span_inside_input_ids=None,
        span_inside_attention_mask=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = self.config.use_return_dict

        outputs = self.bert(
            span_inside_input_ids,
            attention_mask=span_inside_attention_mask,
            # token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class SpanInsideClassificationDataTrainingArguments(
    SpanClassifierDataTrainingArguments
):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    # overwrite_cache: bool = field(
    #     default=False,
    #     metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    # )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )


@dataclass
class SpanInsideClassificationModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    saved_param_path: Optional[str] = field(
        default=None,
        metadata={"help": "Fine-Tuned parameters. If there is, load this parameter."},
    )


from lib.utils.click_based_utils import (
    fix_transformers_argument_type_into_click_argument_type,
)

SpanInsideClassificationModelArguments = (
    fix_transformers_argument_type_into_click_argument_type(
        SpanInsideClassificationModelArguments
    )
)
SpanInsideClassificationDataTrainingArguments = (
    fix_transformers_argument_type_into_click_argument_type(
        SpanInsideClassificationDataTrainingArguments
    )
)
SpanInsideTrainingArguments = fix_transformers_argument_type_into_click_argument_type(
    TrainingArguments
)

from datasets import DatasetDict, Dataset, Sequence, Value, DatasetInfo


class SpanInsideClassifier(Typer):
    def __init__(
        self,
        span_classification_datasets: DatasetDict,
        model_args: SpanInsideClassificationModelArguments,
        data_args: SpanInsideClassificationDataTrainingArguments,
        training_args: TrainingArguments,
    ) -> None:
        # See all possible arguments in src/transformers/training_args.py
        # or by passing the --help flag to this script.
        # We now keep distinct sets of args, for a cleaner separation of concerns.
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
            if is_main_process(training_args.local_rank)
            else logging.WARN,
        )

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()
        logger.info(f"Training/evaluation parameters {training_args}")

        # Set seed before initializing model.
        set_seed(training_args.seed)

        # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
        # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
        # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
        # label if at least two columns are provided.
        #
        # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
        # single column. You can easily tweak this behavior (see below)
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        # if data_args.task_name is not None:
        #     # Downloading and loading a dataset from the hub.
        #     datasets = load_dataset("glue", data_args.task_name)
        # elif data_args.train_file.endswith(".csv"):
        #     # Loading a dataset from local csv files
        #     datasets = load_dataset(
        #         "csv",
        #         data_files={
        #             "train": data_args.train_file,
        #             "validation": data_args.validation_file,
        #         },
        #     )
        # else:
        #     # Loading a dataset from local json files
        #     datasets = load_dataset(
        #         "json",
        #         data_files={
        #             "train": data_args.train_file,
        #             "validation": data_args.validation_file,
        #         },
        #     )
        # See more about loading any type of standard or custom dataset at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Labels
        label_list = span_classification_datasets["train"].features["label"].names
        self.label_list = label_list
        num_labels = len(label_list)

        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            num_labels=num_labels,
            # finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
        model = BertForSpanInsideClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        if model_args.saved_param_path:
            model.load_state_dict(torch.load(model_args.saved_param_path))

        self.tokenizer = tokenizer
        self.model = model

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
            self.max_length = data_args.max_seq_length
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False
            self.max_length = None
        # Preprocessing the datasets
        span_classification_datasets = DatasetDict(
            {
                "train": span_classification_datasets["train"],
                "validation": span_classification_datasets["validation"],
            }
        )
        super().__init__(span_classification_datasets, data_args)
        self.argss += [model_args, data_args, training_args]
        datasets = self.span_classification_datasets

        train_dataset = datasets["train"]
        eval_dataset = datasets["validation"]

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # Get the metric function
        # When datasets metrics include regular accuracy, make an else here and remove special branch from
        # compute_metrics

        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def compute_metrics(p: EvalPrediction):
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            preds = np.argmax(preds, axis=1)
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
            data_collator=default_data_collator
            if data_args.pad_to_max_length
            else None,
        )
        self.trainer = trainer
        # trainer.save_model()
        # Training
        if training_args.do_train:
            trainer.train(
                model_path=model_args.model_name_or_path
                if os.path.isdir(model_args.model_name_or_path)
                else None
            )
            trainer.save_model()  # Saves the tokenizer too for easy upload

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results *****")
                    for key, value in eval_result.items():
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

    def predict(self, tokens: List[str], start: int, end: int) -> SpanClassifierOutput:
        # self.tokenizer()
        tokenized = self.tokenizer(
            [tokens[start:end]],
            is_split_into_words=True,
        )
        args = {
            "span_inside_input_ids": torch.LongTensor(tokenized["input_ids"]).to(
                self.model.device
            ),
            "span_inside_attention_mask": torch.LongTensor(
                tokenized["attention_mask"]
            ).to(self.model.device),
        }
        output = self.model.forward(**args)
        return SpanClassifierOutput(
            label=self.label_list[output.logits.argmax()],
            logits=output.logits.cpu().detach().numpy(),
        )

    def batch_predict(
        self, tokens: List[List[str]], start: List[int], end: List[int]
    ) -> List[SpanClassifierOutput]:
        assert len(tokens) == len(start)
        assert len(start) == len(end)
        # Dataset作成 (datasets.DatasetでOKみたい)
        ## term を取得する
        terms = [ts[s:e] for ts, s, e in zip(tokens, start, end)]
        max_term_len = max(
            len(
                self.tokenizer(
                    term,
                    is_split_into_words=True,
                )["input_ids"]
            )
            for term in terms
        )
        tokenized_terms = self.tokenizer(
            terms,
            padding="max_length",
            max_length=max_term_len,
            # truncation=True,
            is_split_into_words=True,
        )
        dataset = Dataset.from_dict(
            {
                "span_inside_input_ids": tokenized_terms["input_ids"],
                "span_inside_attention_mask": tokenized_terms["attention_mask"],
            }
        )
        outputs = self.trainer.predict(dataset)
        logits = outputs.predictions
        ret_list = []
        for logit in logits:
            ret_list.append(
                SpanClassifierOutput(
                    label=self.label_list[logit.argmax()], logits=logit
                )
            )
        return ret_list

    def preprocess_function(self, example: Dict) -> Dict:
        # Tokenize the texts
        terms = [
            tok[s:e]
            for tok, s, e in zip(example["tokens"], example["start"], example["end"])
        ]
        result = self.tokenizer(
            terms,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            is_split_into_words=True,
        )
        result = {
            "span_inside_input_ids": result["input_ids"],
            "span_inside_attention_mask": result["attention_mask"],
            "label": example["label"],
        }
        return result
