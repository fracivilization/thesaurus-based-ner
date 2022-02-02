import os
import pickle
from typing import List, Optional
import uuid
import numpy as np
from transformers.trainer_utils import set_seed
from src.ner_model.chunker.abstract_model import Chunker
from src.ner_model.multi_label.ml_typer.abstract import (
    MultiLabelTyper,
    MultiLabelTyperConfig,
)
from src.ner_model.typer.data_translator import (
    MSCConfig,
    translate_into_msc_datasets,
    log_label_ratio,
)

from src.utils.hydra import (
    HydraAddaptedTrainingArguments,
    get_orig_transoformers_train_args_from_hydra_addapted_train_args,
)
from .abstract import MultiLabelTyper, MultiLabelTyperOutput, MultiLabelTyperOutput
from dataclasses import field, dataclass
from omegaconf import MISSING
from transformers import TrainingArguments, training_args
from datasets import DatasetDict, Dataset
from loguru import logger
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    set_seed,
)
import torch
import torch.nn
from transformers.models.bert.modeling_bert import BertForTokenClassification
from tqdm import tqdm
from typing import Dict
import itertools
from hashlib import md5
from hydra.utils import get_original_cwd
from transformers.modeling_outputs import (
    TokenClassifierOutput,
)
from scipy.special import softmax
import psutil
import itertools
import datasets
from datasets import DatasetDict
from hydra.utils import get_original_cwd, to_absolute_path

span_start_token = "[unused1]"
span_end_token = "[unused2]"


@dataclass
class MultiLabelEnumeratedModelArguments:
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
    o_label_id: int = 0  # Update when model loading, so 0 is temporal number
    o_sampling_ratio: float = 0.3  # O sampling in train
    nc_sampling_ratio: float = 1.0  # O sampling in train


class BertForEnumeratedMultiLabelTyper(BertForTokenClassification):
    @classmethod
    def from_pretrained(
        cls,
        model_args: MultiLabelEnumeratedModelArguments,
        nc_ids: List[int],
        *args,
        **kwargs,
    ):
        cls.model_args = model_args
        self = super().from_pretrained(model_args.model_name_or_path, *args, **kwargs)
        self.model_args = model_args
        self.nc_ids = nc_ids
        return self

    def __init__(self, config):
        super().__init__(config)
        self.start_classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.end_classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def get_valid_entities(self, starts, ends, labels):
        """
        Get valid entities from start, end and label, cut tensor by no-ent label.
        """
        max_filled_ent_id = (labels != -1).sum(dim=1).max()
        valid_labels = labels[:, :max_filled_ent_id]
        valid_starts = starts[:, :max_filled_ent_id]
        valid_ends = ends[:, :max_filled_ent_id]
        return valid_starts, valid_ends, valid_labels

    def under_sample_o(self, starts, ends, labels, label_masks):
        """
        Get valid entities from start, end and label, cut tensor by no-ent label.
        """
        # O ラベルをサンプリングする
        o_label_mask = labels == self.model_args.o_label_id
        sample_mask = (
            torch.rand(labels.shape, device=labels.device)
            >= self.model_args.o_sampling_ratio
        )  # 1-sample ratio の割合で True となる mask
        sampled_labels = torch.where(o_label_mask * sample_mask, -1, labels)
        # サンプリングに合わせて、-1に当たる部分をソートして外側に出す
        sort_arg = torch.argsort(sampled_labels, descending=True, dim=1)
        sorted_sampled_labels = torch.take_along_dim(sampled_labels, sort_arg, dim=1)
        sorted_starts = torch.take_along_dim(starts, sort_arg, dim=1)
        sorted_ends = torch.take_along_dim(ends, sort_arg, dim=1)
        return sorted_starts, sorted_ends, sorted_sampled_labels

    def under_sample_nc(self, starts, ends, labels):
        """
        Get valid entities from start, end and label, cut tensor by no-ent label.
        """
        # O ラベルをサンプリングする
        nc_ids = torch.cuda.LongTensor(self.nc_ids, device=labels.device)
        nc_label_mask = torch.isin(labels, nc_ids)
        sample_mask = (
            torch.rand(labels.shape, device=labels.device)
            >= self.model_args.nc_sampling_ratio
        )  # 1-sample ratio の割合で True となる mask
        sampled_labels = torch.where(nc_label_mask * sample_mask, -1, labels)
        # サンプリングに合わせて、-1に当たる部分をソートして外側に出す
        sort_arg = torch.argsort(sampled_labels, descending=True, dim=1)
        sorted_sampled_labels = torch.take_along_dim(sampled_labels, sort_arg, dim=1)
        sorted_starts = torch.take_along_dim(starts, sort_arg, dim=1)
        sorted_ends = torch.take_along_dim(ends, sort_arg, dim=1)
        return sorted_starts, sorted_ends, sorted_sampled_labels

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        starts=None,
        ends=None,
        labels=None,
        label_masks=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        if self.training:
            starts, ends, labels = self.under_sample_o(
                starts, ends, labels, label_masks
            )
            # starts, ends, labels = self.get_valid_entities(starts, ends, labels)
        minibatch_size, max_seq_lens = input_ids.shape
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (BatchSize, SeqLength, EmbedDim)
        # device = starts.device
        droped_hidden_states = self.dropout(hidden_states)
        start_logits = self.start_classifier(
            droped_hidden_states
        )  # (BatchSize, SeqLength, ClassNum)
        end_logits = self.end_classifier(
            droped_hidden_states
        )  # (BatchSize, SeqLength, ClassNum)
        starts = starts  # (BatchSize, SeqLength, SpanNum)
        minibatch_size, max_span_num = starts.shape
        start_logits_per_span = torch.gather(
            start_logits,
            1,
            starts[:, :, None].expand(
                minibatch_size, max_span_num, self.config.num_labels
            ),
        )  # (BatchSize, SpanNum, ClassNum)
        # starts_logits_per_span[i,j,k] = start_logits[i,starts[i,j,k],k]
        end_logits_per_span = torch.gather(
            end_logits,
            1,
            ends[:, :, None].expand(
                minibatch_size, max_span_num, self.config.num_labels
            ),
        )  # (BatchSize, SpanNum, ClassNum)
        # ends_logits_per_span[i,j,k] = ends_logits[i,starts[i,j,k],k]
        logits = start_logits_per_span + end_logits_per_span
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.reshape(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class MultiLabelEnumeratedDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    max_span_num: int = field(
        default=512,
        metadata={"help": "Max span num in training."},
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
class MultiLabelEnumeratedTrainingArguments(TrainingArguments):
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."
        },
    )


@dataclass
class MultiLabelEnumeratedTyperConfig(MultiLabelTyperConfig):
    multi_label_typer_name: str = "MultiLabelEnumeratedTyper"
    label_names: str = "non_initialized: this variable is dinamically decided"
    train_datasets: str = "Please add path to DatasetDict"
    model_args: MultiLabelEnumeratedModelArguments = MultiLabelEnumeratedModelArguments(
        model_name_or_path="dmis-lab/biobert-base-cased-v1.1"
    )
    data_args: MultiLabelEnumeratedDataTrainingArguments = (
        MultiLabelEnumeratedDataTrainingArguments()
    )
    train_args: HydraAddaptedTrainingArguments = HydraAddaptedTrainingArguments(
        output_dir="."
    )
    # msc_args: MSCConfig = MSCConfig()


class MultiLabelEnumeratedTyper(MultiLabelTyper):
    def __init__(
        self,
        config: MultiLabelEnumeratedTyperConfig,
    ) -> None:
        """[summary]

        Args:
            span_classification_datasets (DatasetDict): with context tokens
        """
        train_args = get_orig_transoformers_train_args_from_hydra_addapted_train_args(
            config.train_args
        )
        model_args = config.model_args
        data_args = config.data_args
        self.model_args = config.model_args
        self.data_args = config.data_args
        self.training_args = train_args
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
        # msml: Multi Span Multi Label
        msml_datasets = DatasetDict.load_from_disk(
            os.path.join(get_original_cwd(), config.train_datasets)
        )
        log_label_ratio(msml_datasets)
        if train_args.do_train:
            column_names = msml_datasets["train"].column_names
            features = msml_datasets["train"].features
        else:
            column_names = msml_datasets["validation"].column_names
            features = msml_datasets["validation"].features
        # text_column_name = "tokens" if "tokens" in column_names else column_names[0]
        # label_column_name = (
        #     f"{data_args.task_name}_tags"
        #     if f"{data_args.task_name}_tags" in column_names
        #     else column_names[1]
        # )

        label_list = features["labels"].feature.feature.names
        self.label_names = label_list
        num_labels = len(label_list)
        self.label_num = num_labels
        nc_ids = [
            i for i, label in enumerate(self.label_names) if label.startswith("nc-")
        ]

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
            model_args=model_args,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            additional_special_tokens=[span_start_token, span_end_token],
        )
        if "nc-O" in label_list:
            model_args.o_label_id = label_list.index("nc-O")
        else:
            model_args.o_label_id = -2
        model = BertForEnumeratedMultiLabelTyper.from_pretrained(
            model_args,
            nc_ids=nc_ids,
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

        msml_datasets = DatasetDict(
            {
                "train": msml_datasets["train"],
                "validation": msml_datasets["validation"],
            }
        )
        # msml_datasets = DatasetDict(
        #     {
        #         "train": Dataset.from_dict(
        #             msml_datasets["train"][:1500],
        #             features=msml_datasets["train"].features,
        #         ),
        #         "validation": Dataset.from_dict(
        #             msml_datasets["validation"][:1500],
        #             features=msml_datasets["validation"].features,
        #         ),
        #     }
        # )

        def get_buffer_path(arg_str: str):

            return os.path.join(
                get_original_cwd(), "data/buffer", md5(arg_str.encode()).hexdigest()
            )

        cache_file_names = {
            k: get_buffer_path(v._fingerprint) for k, v in msml_datasets.items()
        }
        buffer_file = to_absolute_path(
            "data/buffer/%s" % md5(str(cache_file_names).encode()).hexdigest()
        )
        if not os.path.exists(buffer_file):
            self.span_classification_datasets: DatasetDict = msml_datasets.map(
                self.preprocess_function,
                batched=True,
                load_from_cache_file=True,
                keep_in_memory=True,
                # cache_file_names=cache_file_names,
                num_proc=psutil.cpu_count(logical=False),
                features=datasets.Features(
                    {
                        "input_ids": datasets.Sequence(
                            datasets.Value("int32"),
                            length=config.max_position_embeddings,
                        ),
                        "attention_mask": datasets.Sequence(
                            datasets.Value("int8"),
                            length=config.max_position_embeddings,
                        ),
                        "starts": datasets.Sequence(datasets.Value("int16")),
                        "ends": datasets.Sequence(datasets.Value("int16")),
                        "labels": datasets.features.Array2D(
                            (data_args.max_span_num, num_labels), "bool"
                        ),
                        "label_masks": datasets.Sequence(
                            datasets.Value("bool"),
                            length=config.max_position_embeddings,
                        ),
                        "tokens": datasets.Sequence(datasets.Value("string")),
                    }
                ),
            )
            self.span_classification_datasets.save_to_disk(buffer_file)
        self.span_classification_datasets = DatasetDict.load_from_disk(buffer_file)
        assert (
            len(self.span_classification_datasets["train"][0]["labels"])
            == data_args.max_span_num
        )

        tokenized_datasets = self.span_classification_datasets

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
            args=train_args,
            train_dataset=tokenized_datasets["train"] if train_args.do_train else None,
            eval_dataset=tokenized_datasets["validation"]
            if train_args.do_eval
            else None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
        )
        self.trainer = trainer

    def predict(
        self, tokens: List[str], starts: List[str], ends: List[str]
    ) -> MultiLabelTyperOutput:
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
        # TODO: change into TyperOutput
        raise NotImplementedError
        # return MultiSpanClassifierOutput(
        #     label=self.label_list[outputs.logits[0].argmax()],
        #     logits=outputs.logits[0].cpu().detach().numpy(),
        # )

    # def batch_predict(
    #     self, tokens: List[List[str]], starts: List[int], ends: List[int]
    # ) -> List[TyperOutput]:
    #     assert len(tokens) == len(starts)
    #     assert len(starts) == len(ends)
    #     max_context_len = max(
    #         len(
    #             self.tokenizer(
    #                 tok,
    #                 is_split_into_words=True,
    #             )["input_ids"]
    #         )
    #         for tok in tqdm(tokens)
    #     )
    #     max_span_num = max(len(snt) for snt in starts)
    #     model_input = self.load_model_input(
    #         tokens, starts, ends, max_span_num=max_span_num
    #     )
    #     del model_input["labels"]
    #     dataset = Dataset.from_dict(model_input)
    #     outputs = self.trainer.predict(dataset)
    #     logits = outputs.predictions
    #     ret_list = []
    #     assert all(len(s) == len(e) for s, e in zip(starts, ends))
    #     for logit, span_num in zip(logits, map(len, starts)):
    #         ret_list.append(
    #             TyperOutput(

    #             )
    #             TyperOutput(
    #                 labels=[
    #                     self.label_list[l] for l in logit[:span_num].argmax(axis=1)
    #                 ],
    #                 logits=logit[:span_num],
    #             )
    #         )
    #     return ret_list

    def batch_predict(
        self, tokens: List[List[str]], starts: List[List[int]], ends: List[List[int]]
    ) -> List[MultiLabelTyperOutput]:
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
            tokens, starts, ends, self.label_num, max_span_num=max_span_num
        )
        del model_input["labels"]
        dataset = Dataset.from_dict(model_input)
        outputs = self.trainer.predict(dataset)
        logits = outputs.predictions
        ret_list = []
        assert all(len(s) == len(e) for s, e in zip(starts, ends))
        for logit, span_num in zip(logits, map(len, starts)):
            logit = logit[:span_num]
            probs = softmax(logit, axis=1)
            ret_list.append(
                MultiLabelTyperOutput(
                    labels=[self.label_names[l] for l in logit.argmax(axis=1)],
                    max_probs=probs.max(axis=1),
                    probs=probs,
                )
            )
        return ret_list

    def load_model_input(
        self,
        tokens: List[List[str]],
        snt_starts: List[List[int]],
        snt_ends: List[List[int]],
        label_num: int,
        labels: List[List[int]] = None,
        max_span_num=10000,
    ):
        args = (
            tokens,
            snt_starts,
            snt_ends,
            labels,
            self.data_args,
            max_span_num,
            label_num,
        )
        buffer_file = os.path.join(
            get_original_cwd(), "data/buffer", md5(str(args).encode()).hexdigest()
        )
        if not os.path.exists(buffer_file):
            example = self.padding_spans(
                {
                    "tokens": tokens,
                    "starts": snt_starts,
                    "ends": snt_ends,
                    "labels": labels,
                },
                max_span_num=max_span_num,
            )
            all_padded_subwords = []
            all_attention_mask = []
            all_starts = []
            all_ends = []
            starts = np.array(example["starts"])
            ends = np.array(example["ends"])
            snt_lens = [len(snt) for snt in example["tokens"]]
            snt_split = list(itertools.accumulate(snt_lens))
            all_words = [w for snt in example["tokens"] for w in snt]
            tokenized_tokens = self.tokenizer.batch_encode_plus(
                all_words, add_special_tokens=False
            )["input_ids"]
            snt_split = list(zip([0] + snt_split, snt_split))
            pad_token_id = self.tokenizer.pad_token_id
            for snt_id in tqdm(list(range(len(example["tokens"])))):
                snt_starts = starts[snt_id]
                snt_ends = ends[snt_id]
                s, e = snt_split[snt_id]
                tokens = tokenized_tokens[s:e]
                subwords = [w for li in tokens for w in li]
                # subword2token = list(
                #     itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)])
                # )
                # token2subword = np.array([0] + list(
                #     itertools.accumulate(len(li) for li in tokens)
                # ))
                token2subword = [0] + list(
                    itertools.accumulate(len(li) for li in tokens)
                )
                # new_starts = token2subword[snt_starts]
                # new_ends = token2subword@snt_ends]
                new_starts = [token2subword[s] for s in snt_starts]
                new_ends = [token2subword[e] for e in snt_ends]
                for i, (s, e) in enumerate(zip(new_starts, new_ends)):
                    if s > self.data_args.max_length or e > self.data_args.max_length:
                        pass
                        new_starts[i] = 0
                        new_ends[i] = 0
                assert all(s <= self.data_args.max_length for s in new_starts)
                assert all(e <= self.data_args.max_length for e in new_ends)
                all_starts.append(new_starts)
                all_ends.append(new_ends)
                # token_ids = [sw for word in subwords for sw in word]
                padded_subwords = (
                    [self.tokenizer.cls_token_id]
                    + subwords[: self.data_args.max_length - 2]
                    + [self.tokenizer.sep_token_id]
                )
                all_attention_mask.append(
                    [1] * len(padded_subwords)
                    + [0] * (self.data_args.max_length - len(padded_subwords))
                )
                padded_subwords = padded_subwords + [pad_token_id] * (
                    self.data_args.max_length - len(padded_subwords)
                )
                all_padded_subwords.append(padded_subwords)
                # TODO: labelsをTensor(shapeが一貫したものに変換する)
            all_labels = []
            all_label_masks = []
            padded_labels = [False for i in range(label_num)]
            for snt_labels in example["labels"]:
                ret_snt_labels = []
                ret_snt_label_mask = []
                for span_labels in snt_labels:
                    if -1 in span_labels:
                        ret_snt_labels.append(padded_labels)
                        ret_snt_label_mask.append(False)
                    else:
                        ret_snt_labels.append(
                            [i in span_labels for i in range(label_num)]
                        )
                        ret_snt_label_mask.append(True)
                all_labels.append(ret_snt_labels)
                all_label_masks.append(ret_snt_label_mask)
            ret_dict = {
                "input_ids": all_padded_subwords,
                "attention_mask": all_attention_mask,
                "starts": all_starts,
                "ends": all_ends,
                "labels": np.array(all_labels),
                "label_masks": all_label_masks,
            }
            with open(buffer_file, "wb") as f:
                pickle.dump(ret_dict, f)
        with open(buffer_file, "rb") as f:
            ret_dict = pickle.load(f)
        return ret_dict

    def preprocess_function(self, example: Dict) -> Dict:
        return self.load_model_input(
            example["tokens"],
            example["starts"],
            example["ends"],
            label_num=len(self.label_names),
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
        ignore_span_label = [-1] * max_span_num
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
                                snt[:sn] + [ignore_span_label] * (sn - len(snt))
                            )
                    example[key] = new_value
                else:
                    example[key] = value
        return example

    def train(self):
        # Training
        if self.training_args.do_train:
            self.trainer.train(
                model_path=self.model_args.model_name_or_path
                if os.path.isdir(self.model_args.model_name_or_path)
                else None
            )
            self.trainer.save_model()  # Saves the tokenizer too for easy upload
