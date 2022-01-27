from typing import Dict, List, Type
from .abstract import (
    MultiLabelTyper,
    MultiLabelTyperConfig,
    MultiLabelTyperOutput,
)
from datasets import DatasetDict
from src.utils.string_match import ComplexKeywordTyper
from tqdm import tqdm
from dataclasses import dataclass
from omegaconf import MISSING
from src.dataset.term2cat.term2cat import Term2CatConfig, load_term2cat
import numpy as np
import pickle
from hydra.utils import get_original_cwd
import os
from hydra.utils import to_absolute_path

np.ones(13)


@dataclass
class MultiLabelDictMatchTyperConfig(MultiLabelTyperConfig):
    multi_label_typer_name: str = "MultiLabelDictMatchTyper"
    # term2cat: Term2CatConfig = Term2CatConfig()
    term2cats: str = MISSING  # path for picled term2cat
    label_names: str = "non_initialized"  # this variable is dinamically decided
    # focus_cats: str = MISSING


class MultiLabelDictMatchTyper(MultiLabelTyper):
    def __init__(self, conf: MultiLabelDictMatchTyperConfig) -> None:
        self.conf = conf
        # term2catを無理やり作る(複数カテゴリを連結した文字列をTypeにしてしまえばそれで事足りそう)
        with open(to_absolute_path(conf.term2cats), "rb") as f:
            self.term2cats = pickle.load(f)
        self.keyword_processor = ComplexKeywordTyper(self.term2cats)
        jointed_label_names = set(self.term2cats.values())
        self.label_names = ["nc-O"] + sorted(
            set(
                [
                    label
                    for joint_label in jointed_label_names
                    for label in joint_label.split("_")
                ]
            )
        )

    def predict(
        self, tokens: List[str], starts: List[str], ends: List[str]
    ) -> MultiLabelTyperOutput:
        assert len(starts) == len(ends)
        if not starts:
            return MultiLabelTyperOutput(labels=[], logits=np.array([]))
        else:
            labels = []
            for start, end in zip(starts, ends):
                term = " ".join(tokens[start:end])
                label = self.keyword_processor.type_chunk(term)
                labels.append(label.split("_"))
            # label_ids = np.array([self.label_names.index(l) for l in labels])
            logits = []
            for span_labels in labels:
                label_ids = np.array(
                    [self.label_names.index(label) for label in span_labels]
                )
                logit = 10 * np.eye(len(self.label_names))[label_ids].sum(axis=0)
                logits.append(logit)
            logits = np.array(logits)
            return MultiLabelTyperOutput(labels=labels, logits=logits)

    def train(self):
        pass
