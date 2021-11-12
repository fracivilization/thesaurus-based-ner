import os
import pickle
from typing import List, Optional
import numpy as np
from transformers.trainer_utils import set_seed
from src.ner_model.typer.data_translator import (
    SpanClassificationDatasetArgs,
    translate_into_msc_datasets,
)

from src.utils.hydra import (
    HydraAddaptedTrainingArguments,
    get_orig_transoformers_train_args_from_hydra_addapted_train_args,
)
from .abstract_model import (
    Typer,
    TyperConfig,
    TyperOutput,
)
from dataclasses import field, dataclass
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

span_start_token = "[unused1]"
span_end_token = "[unused2]"


class AttentionPooler(torch.nn.Module):
    def __init__(self, input_dim, middle_dim) -> None:
        super().__init__()
        self.W = torch.nn.Linear(input_dim, middle_dim, bias=False)
        self.w_z = torch.nn.Linear(middle_dim, 1, bias=False)

    def forward(self, hidden_states, attention_mask):
        mask_offset = torch.where(
            torch.logical_or((attention_mask.sum(dim=1) == 0)[:, None], attention_mask),
            torch.zeros_like(attention_mask.float()),
            torch.ones_like(attention_mask.float()) * float("inf"),
        )
        attention_weight = torch.softmax(
            self.w_z(torch.tanh(self.W(hidden_states)))[:, :, 0] - mask_offset, dim=1
        )
        ret = (hidden_states * attention_weight[:, :, None]).sum(dim=1)
        return ret
        # return torch.where((attention_mask.sum(dim=1) == 0)[:,None], torch.zeros_like(ret), ret)
        # torch.matmul(hidden_states, self.W)


@dataclass
class HAMNERLikeModelArguments:
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
    multi_head: bool = field(
        default=False,
        metadata={"help": "use multi-head attention instead of single-head attention"},
    )
    turn_off_inside: bool = False
    turn_off_context: bool = False
    use_span_relational_position_embedding: bool = False
    relational_position_embedding_dim: int = 128
    o_label_id: int = 0  # Update when model loading, so 0 is temporal number
    o_sampling_ratio: float = 0.3  # O sampling in train


def translate_mask_into_multi_head_attention_mask(mask: torch.Tensor):
    mask = torch.logical_not(
        mask[:, None, :]
        .expand(minibatch_size * max_span_num, self.config.num_labels, max_seq_len)
        .reshape(
            minibatch_size * max_span_num * self.config.num_labels,
            1,
            max_seq_len,
        )
    )
    return mask


class BertForHAMNERLikeSpanClassification(BertForTokenClassification):
    @classmethod
    def from_pretrained(cls, model_args: HAMNERLikeModelArguments, *args, **kwargs):
        cls.model_args = model_args
        self = super().from_pretrained(model_args.model_name_or_path, *args, **kwargs)
        self.model_args = model_args
        return self

    def __init__(self, config):
        super().__init__(config)
        input_vec_num = 1 * (not self.model_args.turn_off_inside) + 2 * (
            not self.model_args.turn_off_context
        )
        assert input_vec_num >= 1
        if self.model_args.use_span_relational_position_embedding:
            self.span_relational_position_embedding = torch.nn.Embedding(
                512 * 2 + 1,
                self.model_args.relational_position_embedding_dim,
                padding_idx=0,
            )
            # 左右511と内側およびpadding index: 0 の分あわせて512*2
            # padding用の問題に対して、1024のindexが入る可能性があるため +1
            attention_dim = (
                config.hidden_size + self.model_args.relational_position_embedding_dim
            )
        else:
            attention_dim = config.hidden_size
        self.classifier = torch.nn.Linear(
            input_vec_num * attention_dim, config.num_labels
        )
        self.start_classifier = torch.nn.Linear(attention_dim, config.num_labels)
        self.end_classifier = torch.nn.Linear(attention_dim, config.num_labels)
        self.config = config
        if self.model_args.multi_head:
            pass
            self.left_context_query = torch.nn.parameter.Parameter(
                torch.Tensor(1, 1, attention_dim)
            )
            self.left_context_attention_pooler = torch.nn.MultiheadAttention(
                attention_dim, config.num_labels
            )
            self.right_context_attention_pooler = torch.nn.MultiheadAttention(
                attention_dim, config.num_labels
            )
            self.inside_attention_pooler = torch.nn.MultiheadAttention(
                attention_dim, config.num_labels
            )
        else:
            if not self.model_args.turn_off_context:
                self.left_context_attention_pooler = AttentionPooler(
                    attention_dim, attention_dim
                )
                self.right_context_attention_pooler = AttentionPooler(
                    attention_dim, attention_dim
                )
            if not self.model_args.turn_off_inside:
                self.inside_attention_pooler = AttentionPooler(
                    attention_dim, attention_dim
                )

    def get_valid_entities(self, starts, ends, labels):
        """
        Get valid entities from start, end and label, cut tensor by no-ent label.
        """
        max_filled_ent_id = (labels != -1).sum(dim=1).max()
        valid_labels = labels[:, :max_filled_ent_id]
        valid_starts = starts[:, :max_filled_ent_id]
        valid_ends = ends[:, :max_filled_ent_id]
        return valid_starts, valid_ends, valid_labels

    def under_sample_o(self, starts, ends, labels):
        """
        Get valid entities from start, end and label, cut tensor by no-ent label.
        """
        # O ラベルをサンプリングする
        o_label_mask = labels == self.model_args.o_label_id
        sample_mask = (
            torch.rand(labels.shape) >= self.model_args.o_sampling_ratio
        )  # 1-sample ratio の割合で True となる mask
        sampled_labels = torch.where(o_label_mask * sample_mask, -1, labels)
        # サンプリングに合わせて、-1に当たる部分をソートして外側に出す
        sort_arg = torch.argsort(sampled_labels, descending=True, dim=1)
        sorted_sampled_labels = torch.take_along_dim(sampled_labels, sort_arg, dim=1)
        sorted_starts = torch.take_along_dim(starts, sort_arg, dim=1)
        sorted_ends = torch.take_along_dim(ends, sort_arg, dim=1)
        return sorted_starts, sorted_ends, sorted_sampled_labels

    def forward(
        self, input_ids=None, attention_mask=None, labels=None, starts=None, ends=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        starts, ends, labels = self.under_sample_o(starts, ends, labels)
        starts, ends, labels = self.get_valid_entities(starts, ends, labels)
        minibatch_size, max_seq_lens = input_ids.shape
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states_dim = hidden_states.shape[-1]
        max_seq_len = input_ids.shape[1]
        # start_vectors = (sequence_output * start_mask).sum(dim=1)
        # end_vectors = (sequence_output * end_mask).sum(dim=1)
        # sequence_output = torch.cat([start_vectors, end_vectors], dim=1)
        device = starts.device
        arange = torch.arange(max_seq_len, device=device)[None, None, :]
        _starts = starts[:, :, None]
        _ends = ends[:, :, None]
        max_span_num = starts.shape[1]
        hidden_states_per_span = (
            hidden_states[:, None, :, :]
            .expand(minibatch_size, max_span_num, max_seq_lens, hidden_states_dim)
            .reshape(minibatch_size * max_span_num, max_seq_lens, hidden_states_dim)
        )
        snt_mask = attention_mask[:, None, :].expand(
            minibatch_size, max_span_num, max_seq_len
        )
        if self.model_args.use_span_relational_position_embedding:
            # (1) startの左側のTensorを作る
            input_relational_index = torch.where(arange < _starts, arange - _starts, 0)
            ## arangeからstart位置を引く？
            input_relational_index = torch.where(
                _ends <= arange, arange - _ends + 1, input_relational_index
            )
            # -の部分をなくすために511を足す
            input_relational_index += 512
            # 文の範囲に制限する
            try:
                assert (input_relational_index >= 0).all()
            except AssertionError:
                pass
            padding_id = 0
            input_relational_index = torch.where(
                snt_mask == 1, input_relational_index, padding_id
            )
            # max_index = input_relational_index.reshape(-1).max()
            span_relational_embedding = self.span_relational_position_embedding(
                input_relational_index
            ).reshape(
                minibatch_size * max_span_num,
                max_seq_lens,
                self.model_args.relational_position_embedding_dim,
            )
            hidden_states_per_span = torch.cat(
                [hidden_states_per_span, span_relational_embedding], dim=-1
            )
            # (2) endの右側のTensorを作る
            ## arangeからend位置を引く
            # start 以下の場合 (1), そうでない場合 zeros の input_relational_index: Tensorをつくる
            # end 以上の場合 (2) そうでない場合 input_relational_indexの input_relational_index : Tensor を作る
            # input_relational_index をself.span_relational_position_embedding で埋め込んで、 relational_position_embeddingsをつくる
            # relational_position_embeddingsをhidden_states_per_spanとconcatしてhidden_states_per_spanを作成する

        mention_vecs = []
        if self.model_args.multi_head:
            raise NotImplementedError
            # left_context_mask = translate_mask_into_multi_head_attention_mask(
            #     left_context_mask
            # )
            # left_context_vec = self.left_context_attention_pooler(
            #     self.left_context_query.expand(
            #         1, minibatch_size * max_span_num, hidden_states_dim
            #     ),
            #     hidden_states_per_span.transpose(0, 1),
            #     hidden_states_per_span.transpose(0, 1),
            #     attn_mask=left_context_mask,
            # )
            pass
        else:
            if not self.model_args.turn_off_context:
                left_context_mask = (arange < _starts).reshape(
                    minibatch_size * max_span_num, max_seq_len
                )
                # 後で文内のみを使うように修正する
                right_context_mask = torch.logical_and(
                    _ends <= arange, snt_mask
                ).reshape(minibatch_size * max_span_num, max_seq_len)
                left_context_vec = self.left_context_attention_pooler(
                    hidden_states_per_span,
                    left_context_mask,
                )
                right_context_vec = self.right_context_attention_pooler(
                    hidden_states_per_span,
                    right_context_mask,
                )
                mention_vecs.append(left_context_vec)
                mention_vecs.append(right_context_vec)
            if not self.model_args.turn_off_inside:
                inside_mask = torch.logical_and(
                    _starts <= arange, arange < _ends
                ).reshape(minibatch_size * max_span_num, max_seq_len)
                inside_vec = self.inside_attention_pooler(
                    hidden_states_per_span,
                    inside_mask,
                )
                mention_vecs.append(inside_vec)

        mention_vector = torch.cat(mention_vecs, dim=1)
        pooled_output = self.dropout(mention_vector)
        logits = self.classifier(pooled_output).reshape(
            minibatch_size, max_span_num, -1
        )
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class HAMNERLikeDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    max_span_num: int = field(
        default=1024,
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
class HAMNERLikeTrainingArguments(TrainingArguments):
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."
        },
    )


@dataclass
class HAMNERConfig(TyperConfig):
    typer_name: str = "HAMNER"
    model_args: HAMNERLikeModelArguments = HAMNERLikeModelArguments(
        model_name_or_path="dmis-lab/biobert-base-cased-v1.1"
    )
    data_args: HAMNERLikeDataTrainingArguments = HAMNERLikeDataTrainingArguments()
    train_args: HydraAddaptedTrainingArguments = HydraAddaptedTrainingArguments(
        output_dir="."
    )
    msc_args: SpanClassificationDatasetArgs = SpanClassificationDatasetArgs()
    pass


class HAMNERTyper(Typer):
    def __init__(
        self,
        config: HAMNERConfig,
        ner_datasets: DatasetDict,
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

        # TODO: translate ner_dataset into span_classification_dataset
        span_classification_datasets = translate_into_msc_datasets(
            ner_datasets, config.msc_args
        )
        datasets = span_classification_datasets
        if train_args.do_train:
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
        model_args.o_label_id = label_list.index("nc-O")
        model = BertForHAMNERLikeSpanClassification.from_pretrained(
            model_args,
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

        def get_buffer_path(arg_str: str):

            return os.path.join(
                get_original_cwd(), "data/buffer", md5(arg_str.encode()).hexdigest()
            )

        cache_file_names = {
            k: get_buffer_path(v._fingerprint)
            for k, v in span_classification_datasets.items()
        }
        self.span_classification_datasets = span_classification_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=True,
            # cache_file_names=cache_file_names,
            num_proc=psutil.cpu_count(logical=False),
        )
        assert (
            len(self.span_classification_datasets["train"]["labels"][0])
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
            tokens, starts, ends, max_span_num=max_span_num
        )
        del model_input["labels"]
        dataset = Dataset.from_dict(model_input)
        outputs = self.trainer.predict(dataset)
        logits = outputs.predictions
        ret_list = []
        assert all(len(s) == len(e) for s, e in zip(starts, ends))
        for logit, span_num in zip(logits, map(len, starts)):
            logit = logit[:span_num]
            ret_list.append(
                TyperOutput(
                    labels=[self.label_list[l] for l in logit.argmax(axis=1)],
                    max_probs=softmax(logit, axis=1).max(axis=1),
                )
            )
        return ret_list

    def load_model_input(
        self,
        tokens: List[List[str]],
        starts: List[List[int]],
        ends: List[List[int]],
        labels: List[List[int]] = None,
        max_span_num=10000,
    ):
        args = (
            tokens,
            starts,
            ends,
            labels,
            max_span_num,
            self.data_args,
            max_span_num,
        )
        buffer_file = os.path.join(
            get_original_cwd(), "data/buffer", md5(str(args).encode()).hexdigest()
        )
        if not os.path.exists(buffer_file):

            example = self.padding_spans(
                {"tokens": tokens, "starts": starts, "ends": ends, "labels": labels},
                max_span_num=max_span_num,
            )
            all_padded_subwords = []
            all_attention_mask = []
            all_starts = []
            all_ends = []
            for snt_id, (snt, starts, ends) in tqdm(
                enumerate(zip(example["tokens"], example["starts"], example["ends"])),
                total=len(example["tokens"]),
            ):
                tokens = [
                    self.tokenizer.encode(w, add_special_tokens=False) for w in snt
                ]
                subwords = [w for li in tokens for w in li]
                # subword2token = list(
                #     itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)])
                # )
                token2subword = [0] + list(
                    itertools.accumulate(len(li) for li in tokens)
                )
                new_starts = [token2subword[s] for s in starts]
                new_ends = [token2subword[e] for e in ends]
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
                padded_subwords = padded_subwords + [self.tokenizer.pad_token_id] * (
                    self.data_args.max_length - len(padded_subwords)
                )
                all_padded_subwords.append(padded_subwords)
                ret_dict = {
                    "input_ids": all_padded_subwords,
                    "attention_mask": all_attention_mask,
                    "starts": all_starts,
                    "ends": all_ends,
                    "labels": example["labels"],
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

    def train(self):
        # Training
        if self.training_args.do_train:
            self.trainer.train(
                model_path=self.model_args.model_name_or_path
                if os.path.isdir(self.model_args.model_name_or_path)
                else None
            )
            self.trainer.save_model()  # Saves the tokenizer too for easy upload


class HAMNERLikeSpanContextClassifier(Typer):
    def __init__(
        self,
        span_classification_datasets: DatasetDict,
        model_args: HAMNERLikeModelArguments,
        data_args: HAMNERLikeDataTrainingArguments,
        training_args: HAMNERLikeTrainingArguments,
    ) -> None:
        """[summary]

        Args:
            span_classification_datasets (DatasetDict): with context tokens
        """
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
        model = BertForHAMNERLikeSpanContextClassification.from_pretrained(
            model_args,
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
        super().__init__(span_classification_datasets, data_args)
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
            label=self.label_list[outputs.logits[0].argmax()],
            logits=outputs.logits[0].cpu().detach().numpy(),
        )

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
                [1] * len(padded_subwords)
                + [0] * (self.data_args.max_length - len(padded_subwords))
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
