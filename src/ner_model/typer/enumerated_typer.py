import os
from typing import List, Optional
import numpy as np

from transformers.trainer_utils import set_seed

from src.ner_model.typer.data_translator import (
    SpanClassificationDatasetArgs,
    translate_into_msc_datasets,
)
from .abstract_model import (
    Typer,
    TyperConfig,
    TyperOutput,
)
from dataclasses import field, dataclass

# from transformers import TrainingArguments
from src.utils.hydra import (
    HydraAddaptedTrainingArguments,
    get_orig_transoformers_train_args_from_hydra_addapted_train_args,
)
from datasets import DatasetDict, Dataset
from loguru import logger
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    set_seed,
)
from transformers.modeling_utils import ModelOutput
import torch
from transformers.models.bert.modeling_bert import BertForTokenClassification
from transformers.modeling_outputs import (
    TokenClassifierOutput,
)
from tqdm import tqdm
from typing import Dict
import itertools

span_start_token = "[unused1]"
span_end_token = "[unused2]"


class BertForEnumeratedSpanClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = torch.nn.Linear(2 * config.hidden_size, config.num_labels)
        self.start_classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.end_classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, starts=None, ends=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        # return_dict = self.config.use_return_dict

        minibatch_size, max_seq_lens = input_ids.shape
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states_dim = hidden_states.shape[-1]
        # start_vectors = (sequence_output * start_mask).sum(dim=1)
        # end_vectors = (sequence_output * end_mask).sum(dim=1)
        # sequence_output = torch.cat([start_vectors, end_vectors], dim=1)

        start_positions = starts.reshape(starts.shape + (1,)).expand(
            starts.shape + (hidden_states_dim,)
        )
        start_hidden_states = torch.gather(hidden_states, -2, start_positions)

        end_positions = ends.reshape(ends.shape + (1,)).expand(
            ends.shape + (hidden_states_dim,)
        )
        end_hidden_states = torch.gather(hidden_states, -2, end_positions)

        # start_logits = self.start_classifier(self.dropout(start_hidden_states))
        # end_logits = self.end_classifier(self.dropout(end_hidden_states))

        mention_vector = torch.cat([start_hidden_states, end_hidden_states], dim=-1)
        pooled_output = self.dropout(mention_vector)
        logits = self.classifier(pooled_output)
        # logits = start_logits + end_logits
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            # loss = loss_fct(
            #     start_logits.view(-1, self.num_labels), labels.view(-1)
            # ) + loss_fct(end_logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class EnumeratedTyperModelArguments:
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
class EnumerateTyperDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    max_span_num: int = field(
        default=15,
        metadata={"help": "Max sequence length in training."},
    )
    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
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


@dataclass
class EnumeratedTyperConfig(TyperConfig):
    typer_name: str = "Enumerated"
    train_args: HydraAddaptedTrainingArguments = HydraAddaptedTrainingArguments(
        output_dir="."
    )
    data_args: EnumerateTyperDataTrainingArguments = (
        EnumerateTyperDataTrainingArguments()
    )
    model_args: EnumeratedTyperModelArguments = EnumeratedTyperModelArguments(
        model_name_or_path="dmis-lab/biobert-base-cased-v1.1"
    )
    msc_args: SpanClassificationDatasetArgs = SpanClassificationDatasetArgs()


class EnumeratedTyper(Typer):
    def __init__(
        self,
        conf: EnumeratedTyperConfig,
        ner_datasets: DatasetDict,
    ) -> None:
        """[summary]

        Args:
            span_classification_datasets (DatasetDict): with context tokens
        """
        model_args = conf.model_args
        data_args = conf.data_args
        training_args = (
            get_orig_transoformers_train_args_from_hydra_addapted_train_args(
                conf.train_args
            )
        )
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        logger.info("Start Loading BERT")
        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty."
                "Use --overwrite_output_dir to overcome."
            )
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info("Training/evaluation parameters %s", training_args)
        # Set seed before initializing model.
        set_seed(training_args.seed)

        span_classification_datasets = translate_into_msc_datasets(
            ner_datasets, conf.msc_args
        )

        datasets = span_classification_datasets
        datasets = DatasetDict(
            {"train": datasets["train"], "validation": datasets["validation"]}
        )
        if training_args.do_train:
            column_names = datasets["train"].column_names
            features = datasets["train"].features
        else:
            column_names = datasets["validation"].column_names
            features = datasets["validation"].features
        # text_column_name = "tokens" if "tokens" in column_names else column_names[0]
        # label_column_name = (
        #     f"{data_args.task_name}_tags"
        #     if f"{data_args.task_name}_tags" in column_names
        #     else column_names[1]
        # )

        label_list = features["labels"].feature.names
        self.label_list = label_list
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
            additional_special_tokens=[span_start_token, span_end_token],
        )
        model = BertForEnumeratedSpanClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        if model_args.saved_param_path:
            model.load_state_dict(torch.load(model_args.saved_param_path))
        # Preprocessing the dataset
        # Padding strategy
        self.tokenizer = tokenizer
        self.model = model

        span_classification_datasets = DatasetDict(
            {
                "train": span_classification_datasets["train"],
                "validation": span_classification_datasets["validation"],
            }
        )
        tokenized_datasets = span_classification_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # Data collator
        # data_collator = DataCollatorForTokenClassification(tokenizer)

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
            from seqeval.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            return {
                "accuracy_score": accuracy_score(true_labels, true_predictions),
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
            }

        # Initialize our Trainer
        from transformers import default_data_collator

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"]
            if training_args.do_train
            else None,
            eval_dataset=tokenized_datasets["validation"]
            if training_args.do_eval
            else None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
        )
        self.trainer = trainer

        # Training
        if training_args.do_train:
            trainer.train(
                model_path=model_args.model_name_or_path
                if os.path.isdir(model_args.model_name_or_path)
                else None
            )
            trainer.save_model()  # Saves the tokenizer too for easy upload

    def predict(
        self, tokens: List[str], starts: List[str], ends: List[str]
    ) -> TyperOutput:
        context_tokens = self.get_spanned_token(tokens, starts, ends)
        tokenized_context = self.tokenizer(
            context_tokens,
            # padding="max_length",
            # truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            # max_length=self.data_args.max_length,
            is_split_into_words=True,
            return_offsets_mapping=True,
        )
        tokenized_context
        kwargs = {
            "span_context_input_ids": torch.LongTensor(
                [tokenized_context["input_ids"]]
            ).to(self.model.device),
            "span_context_attention_mask": torch.LongTensor(
                [tokenized_context["attention_mask"]]
            ).to(self.model.device),
        }
        outputs = self.model(**kwargs)
        return TyperOutput(
            labels=self.label_list[outputs.logits[0].argmax()],
            logits=outputs.logits[0].cpu().detach().numpy(),
        )

    # def get_spanned_token(self, tokens: List[str], start: int, end: int):
    #     return (
    #         tokens[:start]
    #         + [span_start_token]
    #         + tokens[start:end]
    #         + [span_end_token]
    #         + tokens[end:]
    #     )

    # def get_spanned_tokens(
    #     self, tokens: List[List[str]], start: List[int], end: List[int]
    # ):
    #     context_tokens = [
    #         self.get_spanned_token(tok, s, e) for tok, s, e in zip(tokens, start, end)
    #     ]
    #     return context_tokens

    def batch_predict(
        self, tokens: List[List[str]], starts: List[List[int]], ends: List[List[int]]
    ) -> List[TyperOutput]:
        assert len(tokens) == len(starts)
        assert len(starts) == len(ends)
        max_context_len = max(
            len(
                self.tokenizer(
                    tok,
                    is_split_into_words=True,
                )["input_ids"]
            )
            for tok in tqdm(tokens)
        )
        max_span_num = max(len(snt) for snt in starts)
        model_input = self.load_model_input(
            tokens, starts, ends, max_length=max_context_len, max_span_num=max_span_num
        )
        del model_input["labels"]
        dataset = Dataset.from_dict(model_input)
        outputs = self.trainer.predict(dataset)
        logits = outputs.predictions
        ret_list = []
        assert all(len(s) == len(e) for s, e in zip(starts, ends))
        for logit, span_num in zip(logits, map(len, starts)):
            ret_list.append(
                TyperOutput(
                    labels=[
                        self.label_list[l] for l in logit[:span_num].argmax(axis=1)
                    ],
                    logits=logit[:span_num],
                )
            )
        return ret_list

    def load_model_input(
        self,
        tokens: List[List[str]],
        starts: List[List[int]],
        ends: List[List[int]],
        max_length: int,
        labels: List[List[int]] = None,
        max_span_num=10000,
    ):
        pass
        example = self.padding_spans(
            {"tokens": tokens, "starts": starts, "ends": ends, "labels": labels},
            max_span_num=max_span_num,
        )
        all_padded_subwords = []
        all_attention_mask = []
        all_starts = []
        all_ends = []
        for snt, starts, ends in zip(
            example["tokens"], example["starts"], example["ends"]
        ):
            tokens = [self.tokenizer.encode(w, add_special_tokens=False) for w in snt]
            subwords = [w for li in tokens for w in li]
            # subword2token = list(
            #     itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)])
            # )
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
            all_starts.append([token2subword[s] for s in starts])
            all_ends.append([token2subword[e] for e in ends])
            # token_ids = [sw for word in subwords for sw in word]
            padded_subwords = (
                [self.tokenizer.cls_token_id]
                + subwords[: self.data_args.max_length - 2]
                + [self.tokenizer.sep_token_id]
            )
            all_attention_mask.append(
                [0] * len(padded_subwords)
                + [1] * (self.data_args.max_length - len(padded_subwords))
            )
            padded_subwords = padded_subwords + [self.tokenizer.pad_token_id] * (
                self.data_args.max_length - len(padded_subwords)
            )
            all_padded_subwords.append(padded_subwords)
        return {
            "input_ids": all_padded_subwords,
            "attention_mask": all_attention_mask,
            "starts": all_starts,
            "ends": all_ends,
            "labels": example["labels"],
        }

    def preprocess_function(self, example: Dict) -> Dict:
        return self.load_model_input(
            example["tokens"],
            example["starts"],
            example["ends"],
            max_length=self.data_args.max_length,
            labels=example["labels"],
            max_span_num=self.data_args.max_span_num,
        )

    def padding_spans(self, example: Dict, max_span_num=10000) -> Dict:
        """add padding for example

        Args:
            example (Dict): {"tokens": List[List[str]], "starts": List[List[int]], "ends": List[List[int]], "label": List[List[int]]}
        Returns:
            Dict: [description]
        """
        sn = max_span_num
        ignore_label = -1
        padding_label = 0
        for key, value in example.items():
            if key in {"starts", "ends", "labels"}:
                new_value = []
                if value:
                    for snt in value:
                        if key in {"starts", "ends"}:
                            new_value.append(
                                snt[:sn] + [padding_label] * (sn - len(snt))
                            )
                        elif key == "labels":
                            new_value.append(
                                snt[:sn] + [ignore_label] * (sn - len(snt))
                            )
                    example[key] = new_value
                else:
                    example[key] = value
        return example
