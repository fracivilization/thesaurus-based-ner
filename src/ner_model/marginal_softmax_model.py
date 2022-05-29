import uuid
from matplotlib.pyplot import axis
from .abstract_model import NERModelConfig, NERModel
from dataclasses import dataclass
from .multi_label.abstract_model import MultiLabelNERModelConfig
from omegaconf.omegaconf import MISSING
from typing import List, Tuple, Optional
import numpy as np
from .multi_label.abstract_model import MultiLabelNERModel
from scipy.special import softmax
from hydra.core.config_store import ConfigStore
from .multi_label import register_multi_label_ner_model, multi_label_ner_model_builder
from datasets import Dataset
from src.dataset.utils import ranked_label2hierarchical_valid_labels
from tqdm import tqdm
import multiprocessing
from p_tqdm import p_map
from more_itertools import chunked


@dataclass
class FlattenMarginalSoftmaxNERModelConfig(NERModelConfig):
    ner_model_name: str = "FlattenMarginalSoftmaxNER"
    multi_label_ner_model: MultiLabelNERModelConfig = MISSING
    focus_cats: str = MISSING
    negative_cats: Optional[str] = None
    hierarchical_valid: bool = False


def postprocess_for_output(example):
    tokens = example["tokens"]
    starts = example["starts"]
    ends = example["ends"]
    outputs = example["outputs"]
    hierarchical_valid: bool = example["hierarchical_valid"]
    label_names: List[str] = example["label_names"]
    focus_cats: List[str] = example["focus_cats"]
    negative_cats: List[str] = example["negative_cats"]
    focus_and_negative_cats: List[str] = example["focus_and_negative_cats"]
    focus_and_negative_label_ids: List[str] = example["focus_and_negative_label_ids"]

    remained_starts = []
    remained_ends = []
    remained_labels = []
    max_probs = []
    for s, e, o in zip(starts, ends, outputs):
        # focus_catsが何かしら出力されているスパンのみ残す
        # 更に残っている場合は最大確率のものを残す
        if hierarchical_valid:
            ranked_labels = [label_names[i] for i in (-o.logits).argsort()]
            valid_labels = ranked_label2hierarchical_valid_labels(ranked_labels)
            focus_labels = set(valid_labels) & set(focus_cats)
            prob = softmax(o.logits)
            if focus_labels:
                assert len(focus_labels) == 1
                label = focus_labels.pop()
            else:
                label = "nc-O"
            remained_labels.append(label)
            focus_and_negative_prob = softmax(o.logits[focus_and_negative_label_ids])
            max_prob = focus_and_negative_prob[focus_and_negative_cats.index(label)]
        else:
            focus_and_negative_prob = softmax(o.logits[focus_and_negative_label_ids])
            max_prob = focus_and_negative_prob.max()
            max_prob = max(focus_and_negative_prob)
            label = focus_and_negative_cats[focus_and_negative_prob.argmax()]
            if label in negative_cats:
                remained_labels.append("nc-%s" % label)
            else:
                remained_labels.append(label)
        remained_starts.append(s)
        remained_ends.append(e)
        max_probs.append(max_prob)
    labeled_chunks = sorted(
        zip(remained_starts, remained_ends, remained_labels, max_probs),
        key=lambda x: x[3],
        reverse=True,
    )
    ner_tags = ["O"] * len(tokens)
    for s, e, label, max_prob in labeled_chunks:
        if not label.startswith("nc-") and all(tag == "O" for tag in ner_tags[s:e]):
            for i in range(s, e):
                if i == s:
                    ner_tags[i] = "B-%s" % label
                else:
                    ner_tags[i] = "I-%s" % label
    assert len(tokens) == len(ner_tags)
    return ner_tags


def batch_postprocess_for_output(examples):
    ner_tags = []
    for example in examples:
        ner_tags.append(postprocess_for_output(example))
    return ner_tags


class FlattenMarginalSoftmaxNERModel(NERModel):
    """Abstract Class for Evaluation"""

    def __init__(self, conf: FlattenMarginalSoftmaxNERModelConfig) -> None:
        self.conf: FlattenMarginalSoftmaxNERModelConfig = conf
        self.multi_label_ner_model: MultiLabelNERModel = multi_label_ner_model_builder(
            conf.multi_label_ner_model
        )  # 後で追加する
        if self.conf.negative_cats:
            self.negative_cats = self.conf.negative_cats.split("_")
        else:
            self.negative_cats = []
        self.focus_cats = self.conf.focus_cats.split("_")
        self.focus_and_negative_cats = (
            ["nc-O"] + self.conf.focus_cats.split("_") + self.negative_cats
        )
        self.focus_and_negative_label_ids = np.array(
            [
                self.multi_label_ner_model.multi_label_typer.label_names.index(label)
                for label in self.focus_and_negative_cats
            ]
        )

    def predict(self, tokens: List[str]) -> List[str]:
        """predict class

        Args:
            sentence (str): input sentence

        Returns:
            List[str]: BIO tags
        """
        raise NotImplementedError

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        """predict class

        Args:
            list of sentence (List[List[str]]): input sentences

        Returns:
            List[List[str]]: BIO tags
        """
        starts, ends, outputs = self.multi_label_ner_model.batch_predict(tokens)
        label_names = self.multi_label_ner_model.label_names
        ner_tags = []
        focus_cats = set(self.focus_and_negative_cats)
        logits = []
        examples = [
            {
                "tokens": snt_tokens,
                "starts": snt_starts,
                "ends": snt_ends,
                "outputs": snt_outputs,
                "hierarchical_valid": self.conf.hierarchical_valid,
                "label_names": label_names,
                "focus_cats": self.focus_cats,
                "negative_cats": self.negative_cats,
                "focus_and_negative_cats": self.focus_and_negative_cats,
                "focus_and_negative_label_ids": self.focus_and_negative_label_ids,
            }
            for snt_tokens, snt_starts, snt_ends, snt_outputs in zip(
                tokens, starts, ends, outputs
            )
        ]
        chunked_examples = list(
            chunked(examples, n=len(examples) // (3 * multiprocessing.cpu_count()))
        )
        ner_tags = p_map(batch_postprocess_for_output, chunked_examples)
        ner_tags = [snt_ner_tags for chunk in ner_tags for snt_ner_tags in chunk]

        for snt_outputs in outputs:
            snt_logits = np.array([o.logits for o in snt_outputs])
            logits.append(snt_logits)
        log_dataset = Dataset.from_dict(
            {
                "tokens": tokens,
                "starts": starts,
                "ends": ends,
                "logits": logits,
                "label_names": [label_names] * len(tokens),
            }
        )
        log_dataset.save_to_disk("span_classif_log_%s" % str(uuid.uuid1()))
        return ner_tags

    def train(self):
        self.multi_label_ner_model.train()


def register_flattern_marginal_softmax_ner_configs(group="ner_model"):
    cs = ConfigStore()
    cs.store(
        group=group,
        name="FlattenMarginalSoftmaxNER",
        node=FlattenMarginalSoftmaxNERModelConfig,
    )
    register_multi_label_ner_model("%s/multi_label_ner_model" % group)
    # raise NotImplementedError
