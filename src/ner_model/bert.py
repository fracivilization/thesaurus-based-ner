import os
from pathlib import Path
from datasets import DatasetDict, ClassLabel
from .abstract_model import NERModel, NERModelConfig
from transformers import pipeline
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    set_seed,
)
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import List
from seqeval.metrics.sequence_labeling import get_entities
from src.utils.params import task_name2ner_label_names
from hashlib import md5
import torch
import numpy as np
from .bond_lib.modeling_roberta import RobertaForTokenClassification
from .bond_lib.modelling_bert import BertForTokenClassification
import re
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import logging

logger = logging.getLogger(__name__)


def accent(text):
    text_a = re.sub(r"à|â|á", "a", text)
    text_i = re.sub(r"ï|î", "i", text_a)
    text_u = re.sub(r"û|ù|ü", "u", text_i)
    text_e = re.sub(r"è|é|ê|ë", "e", text_u)
    text_o = re.sub(r"Ô", "o", text_e)
    text_c = re.sub(r"ç", "c", text_o)
    text_A = re.sub(r"À|Â", "A", text_c)
    text_I = re.sub(r"Ï|Î", "I", text_A)
    text_U = re.sub(r"Û|Ù", "U", text_I)
    text_E = re.sub(r"È|É|Ê|Ë", "E", text_U)
    text_C = re.sub(r"Ç", "C", text_E)
    text_O = re.sub(r"Ô", "O", text_C)
    return text_O


class BERTNERModelBase(NERModel):
    def predict(self, tokens: List[str]) -> List[str]:
        labels = []
        pipe_output = self.pipeline(" ".join(tokens))
        texts = []
        read_tok_id = 0
        for word in pipe_output:
            if isinstance(self.model, BertForTokenClassification):
                text = word["word"].lower()
                if read_tok_id < len(tokens):
                    if texts and tokens[read_tok_id - 1].lower().startswith(
                        texts[-1] + text
                    ):
                        texts[-1] += text
                        continue
                    elif accent(tokens[read_tok_id]).lower().startswith(text):
                        texts.append(text)
                        labels += [
                            self.label_list[int(word["entity"].replace("LABEL_", ""))]
                        ]
                        read_tok_id += 1
                    elif text.startswith("##"):
                        texts[-1] += text[2:]
                    elif text == "-":
                        texts[-1] += text
                    elif texts[-1][-1] == "-":
                        texts[-1] += text
                    else:
                        raise NotImplementedError
            elif isinstance(self.model, RobertaForTokenClassification):
                text = word["word"]
                if text.startswith("Ġ"):
                    texts.append(text[1:])
                    labels += [
                        self.label_list[int(word["entity"].replace("LABEL_", ""))]
                    ]
                    read_tok_id += 1
                else:
                    texts[-1] += text
            else:
                raise NotImplementedError

        assert len(labels) == len(tokens)
        return labels

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        """predict class

        Args:
            list of sentence (List[List[str]]): imput sentences

        Returns:
            List[List[str]]: BIO tags
        """
        # raise NotImplementedError
        return [self.predict(tok) for tok in tokens]


from transformers import TrainingArguments as OrigTrainingArguments
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
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
    saved_param_path: Optional[str] = field(
        default=None,
        metadata={"help": "Fine-Tuned parameters. If there is, load this parameter."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Max sequence length in training."},
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


from omegaconf import MISSING
from transformers.trainer_utils import (
    EvaluationStrategy,
    IntervalStrategy,
    SchedulerType,
    ShardedDDPOption,
)


@dataclass
class TrainingArguments(OrigTrainingArguments):
    eval_steps: Optional[int] = field(
        default=None, metadata={"help": "Run an evaluation every X steps."}
    )
    push_to_hub_organization: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the organization in with to which push the `Trainer`."
        },
    )
    push_to_hub_token: Optional[str] = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."}
    )
    evaluation_strategy: IntervalStrategy = field(
        default=IntervalStrategy.NO,
        metadata={"help": "The evaluation strategy to use."},
    )
    lr_scheduler_type: SchedulerType = field(
        default=SchedulerType.LINEAR,
        metadata={"help": "The scheduler type to use."},
    )
    logging_strategy: IntervalStrategy = field(
        default=IntervalStrategy.STEPS,
        metadata={"help": "The logging strategy to use."},
    )
    save_strategy: IntervalStrategy = field(
        default=IntervalStrategy.STEPS,
        metadata={"help": "The checkpoint save strategy to use."},
    )
    push_to_hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository to which push the `Trainer`."},
    )

    def __post_init__(self):
        """Adapt hydra because some parameters (e.g. debug) is changed in those type
        after init train args by __post_init__ of parent class
        Please pass this "raw" TrainingArguments into original TrainingArguments"""
        pass


@dataclass
class BERTModelConfig(NERModelConfig):
    ner_model_name: str = "BERT"
    train_args: TrainingArguments = TrainingArguments(output_dir=".")
    data_args: DataTrainingArguments = DataTrainingArguments()
    model_args: ModelArguments = ModelArguments(
        model_name_or_path="dmis-lab/biobert-base-cased-v1.1"
    )


def register_BERT_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="ner_model/train_args",
        name="BERT_train_args",
        node=TrainingArguments,
    )
    cs.store(
        group="ner_model/model_args",
        name="BERT_model_args",
        node=ModelArguments,
    )
    cs.store(
        group="ner_model/data_args",
        name="BERT_data_args",
        node=DataTrainingArguments,
    )
    cs.store(group="ner_model", name="base_BERT_model_config", node=BERTModelConfig)


class BERTNERModel(BERTNERModelBase):
    def __init__(self, ner_dataset: DatasetDict, config: BERTModelConfig):
        super().__init__()
        self.conf["config"] = config
        train_args_dict = {k: v for k, v in config.train_args.items() if k != "_n_gpu"}
        train_args = OrigTrainingArguments(**train_args_dict)
        self.train_args = train_args
        data_args = config.data_args
        model_args = config.model_args
        self.model_args = model_args
        self.label_list = ner_dataset["train"].features["ner_tags"].feature.names
        # self.args["training_args"] = training_args
        # self.args["data_args"] = data_args
        # self.args["model_args"] = model_args
        # self.args["task"] = task
        self.datasets_hash = {
            key: split.__hash__() for key, split in ner_dataset.items()
        }
        self.conf["ner_dataset"] = self.datasets_hash
        config.train_args.output_dir = os.path.join(
            "data/output", md5(str(self.conf).encode()).hexdigest()
        )
        logger.info("output dir: %s" % config.train_args.output_dir)
        logger.info("Start Loading BERT")
        if (
            os.path.exists(train_args.output_dir)
            and os.listdir(train_args.output_dir)
            and train_args.do_train
            and not train_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({train_args.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
        logger.warning(
            f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}"
            + f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
        )
        logger.info("Training/evaluation parameters %s", train_args)
        # Set seed before initializing model.
        set_seed(train_args.seed)

        datasets = ner_dataset
        datasets = DatasetDict(
            {"train": datasets["train"], "validation": datasets["validation"]}
        )

        if train_args.do_train:
            column_names = datasets["train"].column_names
            features = datasets["train"].features
        else:
            column_names = datasets["validation"].column_names
            features = datasets["validation"].features
        text_column_name = "tokens" if "tokens" in column_names else column_names[0]
        label_column_name = (
            f"{data_args.task_name}_tags"
            if f"{data_args.task_name}_tags" in column_names
            else column_names[1]
        )

        # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
        # unique labels.
        def get_label_list(labels):
            unique_labels = set()
            for label in labels:
                unique_labels = unique_labels | set(label)
            label_list = list(unique_labels)
            label_list.sort()
            return label_list

        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
            # No need to convert the labels since they are already ints.
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            label_list = get_label_list(datasets["train"][label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}
        num_labels = len(label_list)

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        if model_args.saved_param_path:
            model.load_state_dict(torch.load(model_args.saved_param_path))
        # Preprocessing the dataset
        # Padding strategy
        padding = "max_length" if data_args.pad_to_max_length else False

        # Tokenize all texts and align the labels with them.
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
                return_offsets_mapping=True,
                max_length=data_args.max_length,
            )
            offset_mappings = tokenized_inputs.pop("offset_mapping")
            labels = []
            for label, offset_mapping in zip(
                examples[label_column_name], offset_mappings
            ):
                label_index = 0
                current_label = -100
                label_ids = []
                for offset in offset_mapping:
                    # We set the label for the first token of each word. Special characters will have an offset of (0, 0)
                    # so the test ignores them.
                    if offset[0] == 0 and offset[1] != 0:
                        current_label = label_to_id[label[label_index]]
                        label_index += 1
                        label_ids.append(current_label)
                    # For special tokens, we set the label to -100 so it's automatically ignored in the loss function.
                    elif offset[0] == 0 and offset[1] == 0:
                        label_ids.append(-100)
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(
                            current_label if data_args.label_all_tokens else -100
                        )

                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenized_datasets = datasets.map(
            tokenize_and_align_labels,
            batched=True,
            # num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer)

        # Metrics
        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            return {
                "accuracy_score": accuracy_score(true_labels, true_predictions),
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
            }

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized_datasets["train"] if train_args.do_train else None,
            eval_dataset=tokenized_datasets["validation"]
            if train_args.do_eval
            else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        self.trainer = trainer

        self.model = model
        self.tokenizer = tokenizer
        if isinstance(model.device.index, int):
            self.pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                device=model.device.index,
            )
        else:
            self.pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
            )

    def train(self):
        # Training
        if self.train_args.do_train:
            self.trainer.train(
                model_path=self.model_args.model_name_or_path
                if os.path.isdir(self.model_args.model_name_or_path)
                else None
            )
            self.trainer.save_model()  # Saves the tokenizer too for easy upload

    def predict(self, tokens: List[str]) -> List[str]:
        labels = []
        pipe_output = self.pipeline(" ".join(tokens))
        texts = []
        read_tok_id = 0
        for word in pipe_output:
            text = word["word"].lower()
            if read_tok_id < len(tokens):
                if texts and tokens[read_tok_id - 1].lower().startswith(
                    texts[-1] + text
                ):
                    texts[-1] += text
                    continue
                elif tokens[read_tok_id].lower().startswith(text):

                    texts.append(text)
                    labels += [
                        self.label_list[int(word["entity"].replace("LABEL_", ""))]
                    ]
                    read_tok_id += 1
                elif text.startswith("##"):
                    texts[-1] += text[2:]
                else:
                    raise NotImplementedError

        assert len(labels) == len(tokens)
        return labels

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        """predict class

        Args:
            list of sentence (List[List[str]]): input sentences

        Returns:
            List[List[str]]: BIO tags
        """
        # raise NotImplementedError
        return [self.predict(tok) for tok in tokens]
