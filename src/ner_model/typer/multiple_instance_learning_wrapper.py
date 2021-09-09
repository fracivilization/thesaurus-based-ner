from dataclasses import dataclass
from lib.utils.click_based_utils import (
    fix_transformers_argument_type_into_click_argument_type,
)
from typing import Dict, List
from .abstract_model import Typer, SequenceClassifierOutputPlus
import torch
import torch.nn
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers.data.data_collator import InputDataClass
import inspect
from loguru import logger


@dataclass
class MILWrapperModelArgs:
    attention_embed_dim: int = 128  # 本論文の L


class MILWrapperModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, model_args: MILWrapperModelArgs) -> None:
        super().__init__()
        self.model = model
        self.model_args = model_args
        self.V = torch.nn.Linear(
            self.model.classifier.in_features,
            self.model_args.attention_embed_dim,
            bias=False,
        )
        self.w = torch.nn.Linear(self.model_args.attention_embed_dim, 1, bias=False)
        self.dropout = self.model.dropout
        self.classifier = self.model.classifier

    def forward(self, labels=None, **kwargs):
        minibatch_size, snt_per_bag, max_seq_len = kwargs[
            "span_context_input_ids"
        ].shape
        new_kwargs = dict()
        for key, tensor in kwargs.items():
            new_kwargs[key] = tensor.reshape(
                (minibatch_size * snt_per_bag, max_seq_len)
            )
        outputs: SequenceClassifierOutputPlus = self.model.forward(**new_kwargs)
        feature_vecs = outputs.feature_vecs.reshape(minibatch_size, snt_per_bag, -1)
        attetion_weight = torch.softmax(self.w(torch.tanh(self.V(feature_vecs))), dim=1)
        bag_feature_vecs = (feature_vecs * attetion_weight).sum(dim=1)
        logits = self.classifier(self.dropout(bag_feature_vecs))
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return SequenceClassifierOutputPlus(loss=loss, logits=logits)


from datasets import DatasetDict, Dataset

MILWrapperTrainingArguments = fix_transformers_argument_type_into_click_argument_type(
    TrainingArguments
)


class MILWrapper:
    def __init__(
        self,
        span_classifier: Typer,
        mil_datasets: DatasetDict,
        train_args: MILWrapperTrainingArguments,
        model_args: MILWrapperModelArgs,
    ) -> None:
        self.train_args = train_args
        self.span_classifier = span_classifier
        self.model = MILWrapperModel(span_classifier.model, model_args)
        self.mil_datasets = mil_datasets
        self.preprocessed_mil_datasets = mil_datasets.map(self.preprocess_function)
        trainer = Trainer(
            self.model,
            train_args,
            self.data_collator,
            self.preprocessed_mil_datasets["train"],
            self.preprocessed_mil_datasets["validation"],
        )
        self.trainer = trainer
        if train_args.do_train:
            trainer.train()
            trainer.save_model()

    def preprocess_function(self, example: Dict):
        input_example = dict()
        input_example["tokens"] = example["tokens"]
        input_example["start"] = example["starts"]
        input_example["end"] = example["ends"]
        if "label" in example:
            input_example["label"] = example["label"]
        return self.span_classifier.preprocess_function(input_example)

    def data_collator(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        signature = inspect.signature(self.span_classifier.model.forward)
        signature_columns = list(signature.parameters.keys())
        features
        tensor_dict = dict()
        for key in signature_columns:
            if key == "labels":
                if any("label" in snt for snt in features):
                    tensor_dict[key] = torch.LongTensor([f["label"] for f in features])
            else:
                tensor_dict[key] = torch.LongTensor([f[key] for f in features])
        return tensor_dict


from sklearn.metrics import classification_report, accuracy_score


class MILEvaluator:
    def __init__(self, mil_model: MILWrapper, test_dataset: Dataset) -> None:
        self.mil_model = mil_model
        self.test_dataset = test_dataset
        preprocessed_test_dataset = self.test_dataset.map(mil_model.preprocess_function)
        outputs = self.mil_model.trainer.predict(preprocessed_test_dataset)
        label_names = test_dataset.features["label"].names
        pred_labels = [label_names[l] for l in outputs.predictions.argmax(axis=1)]
        gold_labels = [label_names[l] for l in test_dataset["label"]]
        logger.info(classification_report(gold_labels, pred_labels))
        logger.info("accuracy: %s" % str(accuracy_score(gold_labels, pred_labels)))
        pass