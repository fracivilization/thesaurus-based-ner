from transformers.data.data_collator import InputDataClass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import PrinterCallback
from .span_inside_classifier import SpanInsideClassifier
from .span_context_classifier import SpanContextClassifier
from .abstract_model import (
    Typer,
    SpanClassifierOutput,
    SpanClassifierDataTrainingArguments,
)
from datasets import DatasetDict, Dataset
from typing import Callable, Counter, Dict, List, Optional, Set, Tuple, Union
import os
from transformers.trainer import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    DataCollator,
    EvalPrediction,
    TrainerCallback,
    set_seed,
    default_data_collator,
    DataCollatorWithPadding,
    DEFAULT_CALLBACKS,
    CallbackHandler,
    PrinterCallback,
    is_torch_tpu_available,
    is_datasets_available,
    TrainerState,
    TrainerControl,
    _use_native_amp,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    DEFAULT_PROGRESS_CALLBACK,
)
import torch
from loguru import logger
import collections
import datasets
from dataclasses import dataclass, field
from lib.utils.click_based_utils import (
    fix_transformers_argument_type_into_click_argument_type,
)
from scipy.special import softmax
import numpy as np
from transformers.trainer_callback import TrainerCallback


class CoSpanClassificationTrainerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        kwargs["model"].add_epoch()


@dataclass
class SpanCoTeachingClassifierTrainingArguments(TrainingArguments):
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Remove columns not required by the model when using an nlp.Dataset."
        },
    )


SpanCoTeachingClassifierTrainingArguments = (
    fix_transformers_argument_type_into_click_argument_type(
        SpanCoTeachingClassifierTrainingArguments
    )
)

import inspect
from transformers.modeling_outputs import SequenceClassifierOutput
import click


@dataclass
class CoSpanClassifierModelArguments:
    tau: float = 0.4  # noise rate
    t_k: int = 10  # how many epoch to derease data usage into 1 - tau
    ensemble: click.Choice(["mean", "product"]) = "mean"
    saved_param_path: str = None


@dataclass
class JoCoRClassifierModelArguments(CoSpanClassifierModelArguments):
    λ: float = 0.95


import math


class CoSpanClassifierModel(torch.nn.Module):
    def __init__(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        model_args: CoSpanClassifierModelArguments,
    ) -> None:
        super().__init__()
        self.model1 = model1
        # self.args1 = set(inspect.signature(self.model1.forward).parameters.keys())
        self.args1 = self.get_model_forward_args(self.model1)
        self.model2 = model2
        self.args2 = self.get_model_forward_args(self.model2)
        self.model_args = model_args
        self.epoch = 0
        self.keep_rate = 1

    def add_epoch(self):
        self.epoch += 1
        self.keep_rate = 1 - self.model_args.tau * min(
            self.epoch / self.model_args.t_k, 1
        )

    def forward(self, **kwargs):
        raise NotImplementedError

    def get_model_forward_args(self, model: torch.nn.Module) -> Set:
        if isinstance(model, CoSpanClassifierModel):
            args = set()
            for imo, mo in enumerate([model.model1, model.model2]):
                args |= {
                    "%s_for_span_classifier%d" % (k, imo + 1)
                    for k in self.get_model_forward_args(mo)
                }
            return args
        else:
            return set(inspect.signature(model.forward).parameters.keys())

    def get_each_model_outputs(self, **kwargs):
        kwargs1 = {
            key[:-21]: kwargs[key]
            for key in set(kwargs.keys())
            & {"%s_for_span_classifier1" % k for k in self.args1}
        }
        kwargs2 = {
            key[:-21]: kwargs[key]
            for key in set(kwargs.keys())
            & {"%s_for_span_classifier2" % k for k in self.args2}
        }
        model1_outputs = self.model1(**kwargs1)
        model2_outputs = self.model2(**kwargs2)
        return model1_outputs, model2_outputs


class CoMeanModel(CoSpanClassifierModel):
    def forward(self, **kwargs):
        model1_outputs, model2_outputs = self.get_each_model_outputs(**kwargs)
        labels = kwargs["labels_for_span_classifier1"]
        loss_fct = torch.nn.CrossEntropyLoss()
        if self.model_args.ensemble == "product":
            logits = torch.log_softmax(
                model1_outputs.logits, dim=1
            ) + torch.log_softmax(model2_outputs.logits, dim=1)
        elif self.model_args.ensemble == "mean":
            logits = (
                torch.softmax(model1_outputs.logits, dim=1)
                + torch.softmax(model2_outputs.logits, dim=1)
            ) / 2
        else:
            raise NotImplementedError
        loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class CoTeachingModel(CoSpanClassifierModel):
    def forward(self, **kwargs):
        model1_outputs, model2_outputs = self.get_each_model_outputs(**kwargs)
        labels = [val for key, val in kwargs.items() if key.startswith("labels")][0]
        batch_size = len(labels)
        num_remember = math.floor(batch_size * self.keep_rate)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss1 = loss_fct(model1_outputs.logits, labels)
        loss2 = loss_fct(model2_outputs.logits, labels)
        # 予備実験的に、片方のモデルが難しいものだけ学習させてみる
        remained_loss2 = torch.gather(
            loss2, 0, torch.argsort(loss1)[:num_remember]
        ).mean()
        remained_loss1 = torch.gather(
            loss1, 0, torch.argsort(loss2)[:num_remember]
        ).mean()
        return SequenceClassifierOutput(loss=remained_loss1 + remained_loss2)


import torch.nn.functional as F


def kl_loss_compute(pred, soft_targets, reduction="none"):

    kl = F.kl_div(
        F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduction="none"
    )
    if reduction != "none":
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


class JoCoRModel(CoSpanClassifierModel):
    def __init__(
        self, model1, model2, model_args: JoCoRClassifierModelArguments
    ) -> None:
        super().__init__(model1, model2, model_args)
        self.λ = self.model_args.λ

    def forward(self, **kwargs):
        model1_outputs, model2_outputs = self.get_each_model_outputs(**kwargs)
        labels = kwargs["labels"]
        batch_size = len(labels)
        num_remember = math.floor(batch_size * self.keep_rate)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss1 = loss_fct(model1_outputs.logits, labels)
        loss2 = loss_fct(model2_outputs.logits, labels)
        supervised_loss = loss1 + loss2
        λ = self.λ
        contrastive_loss = kl_loss_compute(
            model1_outputs.logits, model2_outputs.logits
        ) + kl_loss_compute(model2_outputs.logits, model1_outputs.logits)
        loss = (1 - λ) * supervised_loss + λ * contrastive_loss
        return SequenceClassifierOutput(
            loss=loss[torch.argsort(loss)[:num_remember]].mean()
        )


class CoSpanClassifier(Typer):
    def __init__(
        self,
        span_classifier1: Typer,
        span_classifier2: Typer,
        model: CoSpanClassifierModel,
        span_classification_datasets: DatasetDict,
        data_args: SpanClassifierDataTrainingArguments,
        training_args: SpanCoTeachingClassifierTrainingArguments,
        model_args: CoSpanClassifierModelArguments,
    ) -> None:
        self.span_classifier1 = span_classifier1
        self.span_classifier2 = span_classifier2
        self.span_classification_datasets = span_classification_datasets
        self.data_args = data_args
        self.training_args = training_args
        self.model = model
        self.model_args = model_args
        if model_args.saved_param_path:
            self.model.load_state_dict(torch.load(model_args.saved_param_path))
        span_classification_datasets = DatasetDict(
            {
                "train": span_classification_datasets["train"],
                "validation": span_classification_datasets["validation"],
            }
        )
        super().__init__(span_classification_datasets, data_args)
        self.argss += (
            [data_args, training_args, model_args]
            + self.span_classifier1.argss
            + self.span_classifier2.argss
        )
        trainer = Trainer(
            self.model,
            self.training_args,
            self.data_collator,
            self.span_classification_datasets["train"],
            self.span_classification_datasets["validation"],
            callbacks=[CoSpanClassificationTrainerCallback],
        )
        if self.training_args.do_train:
            trainer.train()
            trainer.save_model()
        self.label_list = (
            self.span_classification_datasets["train"].features["label"].names
        )

    def predict(self, tokens: List[str], start: int, end: int) -> SpanClassifierOutput:
        pass

    def preprocess_function(self, example: Dict) -> Dict:
        preprocessed1 = self.span_classifier1.preprocess_function(example)
        preprocessed2 = self.span_classifier2.preprocess_function(example)
        # duplicated_keys = set(preprocessed1.keys()) & set(preprecessed2.keys())
        # for k in duplicated_keys:
        #     assert preprocessed1[k] == preprecessed2[k]
        ret_dict = dict()
        for k, v in preprocessed1.items():
            ret_dict["%s_for_span_classifier1" % k] = v
        for k, v in preprocessed2.items():
            ret_dict["%s_for_span_classifier2" % k] = v
        return ret_dict

    def get_function_args(self, span_classifier) -> Set:
        if isinstance(span_classifier, CoSpanClassifier):
            args = set()
            for isc, sc in enumerate(
                [span_classifier.span_classifier1, span_classifier.span_classifier2]
            ):
                args |= {
                    "%s_for_span_classifier%d" % (k, isc + 1)
                    for k in self.get_function_args(sc)
                }
            return args
        else:
            return set(
                inspect.signature(span_classifier.model.forward).parameters.keys()
            )

    def data_collator(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        args = self.get_function_args(self)
        args = list(args)
        tensor_dict = dict()
        for key in args:
            if key == "labels":
                tensor_dict[key] = torch.LongTensor([f["label"] for f in features])
            elif key.startswith("labels_for_span_classifier"):
                tensor_dict[key] = torch.LongTensor(
                    [f[key.replace("labels", "label")] for f in features]
                )
            else:
                tensor_dict[key] = torch.LongTensor([f[key] for f in features])
        return tensor_dict

    def batch_predict(
        self, tokens: List[List[str]], start: List[int], end: List[int]
    ) -> List[SpanClassifierOutput]:
        output1 = self.span_classifier1.batch_predict(tokens, start, end)
        output2 = self.span_classifier2.batch_predict(tokens, start, end)
        outputs = []
        for o1, o2 in zip(output1, output2):
            prob1 = softmax(o1.logits)
            prob2 = softmax(o2.logits)
            if self.model_args.ensemble == "mean":
                prob = (prob1 + prob2) / 2
            elif self.model_args.ensemble == "product":
                prob = prob1 * prob2
            else:
                raise NotImplementedError
            label = self.label_list[(prob1 + prob2).argmax()]
            outputs.append(SpanClassifierOutput(label, np.log(prob)))
        return outputs


from sklearn.metrics import log_loss
from collections import Counter


def noise_rate_by_label(
    co_teaching_classifier: CoSpanClassifier,
    span_classification_dataset: Dataset,
):
    noise_rate = co_teaching_classifier.model_args.tau
    noise_number = math.ceil(len(span_classification_dataset) * noise_rate)
    label_names = span_classification_dataset.features["label"].names
    original_label_statistics = Counter(
        [label_names[l] for l in span_classification_dataset["label"]]
    )
    logger.info("Original_label_statistics: %s" % original_label_statistics)
    span_classifier: Typer = None
    for classifier_id, span_classifier in enumerate(
        [
            co_teaching_classifier.span_classifier1,
            co_teaching_classifier.span_classifier2,
        ]
    ):
        pass
        outputs = span_classifier.batch_predict(
            span_classification_dataset["tokens"],
            span_classification_dataset["start"],
            span_classification_dataset["end"],
        )
        probs = softmax(np.stack([o.logits for o in outputs]), axis=1)
        probs[np.array(span_classification_dataset["label"])[None]]
        max_probs = (
            probs
            * np.eye(len(span_classification_dataset.features["label"].names))[
                span_classification_dataset["label"]
            ]
        ).sum(axis=1)
        (probs * np.eye(len(label_names))[span_classification_dataset["label"]]).sum(
            axis=1
        )
        log_loss(span_classification_dataset["label"], probs, normalize=False)
        noise_sample_index = np.argsort(max_probs)[:noise_number]
        noise_predicted_label_statistics = Counter(
            [
                label_names[l]
                for l in np.array(span_classification_dataset["label"])[
                    noise_sample_index
                ]
            ]
        )
        logger.info(
            "noise label statistics predicted by classifier %d: %s"
            % (classifier_id, type(span_classifier))
        )
        logger.info(noise_predicted_label_statistics.most_common())
        logger.info(
            "So, remained label statistics for classifier %d" % (1 - classifier_id,)
        )
        logger.info(original_label_statistics - noise_predicted_label_statistics)
        pass
