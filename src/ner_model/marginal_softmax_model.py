import uuid
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
from src.dataset.utils import load_negative_cats_from_positive_cats
import multiprocessing
from p_tqdm import p_map
from more_itertools import chunked
from src.dataset.utils import CoNLL2003CategoryMapper
from collections import defaultdict


@dataclass
class FlattenMarginalSoftmaxNERModelConfig(NERModelConfig):
    ner_model_name: str = "FlattenMarginalSoftmaxNER"
    multi_label_ner_model: MultiLabelNERModelConfig = MISSING
    positive_cats: str = MISSING
    eval_dataset: str = MISSING  # CoNLL2003 or MedMentions
    with_negative_categories: bool = True


def postprocess_for_output(example):
    tokens = example["tokens"]
    starts = example["starts"]
    ends = example["ends"]
    outputs = example["outputs"]
    negative_cats: List[str] = example["negative_cats"]
    used_logit_label_names: List[str] = example["used_logit_label_names"]
    used_logit_label_ids: List[str] = example["used_logit_label_ids"]
    category_mapper: dict = example["category_mapper"]

    category_mapped_focus_and_negative_cats = []
    for cat in used_logit_label_names:
        if cat in category_mapper:
            category_mapped_focus_and_negative_cats.append(category_mapper[cat])
        else:
            category_mapped_focus_and_negative_cats.append(cat)
    category_mapped_focus_and_negative_cats = list(
        sorted(set(category_mapped_focus_and_negative_cats))
    )
    remained_starts = []
    remained_ends = []
    remained_labels = []
    max_probs = []
    for s, e, o in zip(starts, ends, outputs):
        # positive_catsが何かしら出力されているスパンのみ残す
        # 更に残っている場合は最大確率のものを残す
        focus_and_negative_prob = softmax(o.logits[used_logit_label_ids])
        if category_mapper:
            category_mapped_probs = [0] * len(category_mapped_focus_and_negative_cats)
            assert len(used_logit_label_names) == len(focus_and_negative_prob)
            for cat, prob in zip(used_logit_label_names, focus_and_negative_prob):
                if category_mapper:
                    if cat in category_mapper:
                        cat = category_mapper[cat]
                    cat_id = category_mapped_focus_and_negative_cats.index(cat)
                    category_mapped_probs[cat_id] += prob
            used_logit_label_names = category_mapped_focus_and_negative_cats
            focus_and_negative_prob = np.array(category_mapped_probs)
        max_prob = focus_and_negative_prob.max()
        label = used_logit_label_names[focus_and_negative_prob.argmax()]
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
        label_names = self.multi_label_ner_model.multi_label_typer.label_names
        self.positive_cats = [
            cat for cat in self.conf.positive_cats.split("_") if cat in label_names
        ]
        if self.conf.with_negative_categories:
            self.negative_cats = []
            for cat in sorted(
                load_negative_cats_from_positive_cats(
                    self.positive_cats, conf.eval_dataset
                )
            ):
                if cat in label_names:
                    self.negative_cats.append(cat)
        else:
            self.negative_cats = []
        focus_and_negative_cats = ["nc-O"] + self.positive_cats + self.negative_cats
        self.category_mapper = dict()
        used_logit_label_names = []
        used_logit_label_ids = []
        for label in focus_and_negative_cats:
            if label in CoNLL2003CategoryMapper:
                for category in CoNLL2003CategoryMapper[label]:
                    # NOTE: 学習データ中に出現しなかったラベルは無視する
                    self.category_mapper[category] = label
                    used_logit_label_names.append(category)
                    used_logit_label_ids.append(label_names.index(category))
            # NOTE: 学習データ中に出現しなかったラベルは無視する
            else:
                used_logit_label_names.append(label)
                used_logit_label_ids.append(label_names.index(label))
        self.used_logit_label_names = used_logit_label_names
        self.used_logit_label_ids = np.array(used_logit_label_ids)

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
        logits = []
        examples = [
            {
                "tokens": snt_tokens,
                "starts": snt_starts,
                "ends": snt_ends,
                "outputs": snt_outputs,
                "label_names": label_names,
                "positive_cats": self.positive_cats,
                "negative_cats": self.negative_cats,
                "used_logit_label_names": self.used_logit_label_names,
                "used_logit_label_ids": self.used_logit_label_ids,
                "category_mapper": self.category_mapper,
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
