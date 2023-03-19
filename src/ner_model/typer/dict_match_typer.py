from typing import Dict, List, Type
from .abstract_model import (
    Typer,
    SpanClassifierDataTrainingArguments,
    SpanClassifierOutput,
    TyperConfig,
    TyperOutput,
)
from datasets import DatasetDict
from src.utils.string_match import ComplexKeywordTyper
from tqdm import tqdm
from dataclasses import dataclass
from omegaconf import MISSING
import numpy as np
import pickle
from hydra.utils import get_original_cwd
import os

np.ones(13)


@dataclass
class DictMatchTyperConfig(TyperConfig):
    typer_name: str = "DictMatchTyper"
    term2cat: str = MISSING  # path for pickled term2cat
    label_names: str = "non_initialized"  # this variable is dinamically decided
    case_sensitive: bool = (
        False  # Falseであっても、大文字・小文字の違いによってtypeが異なる場合にはcase_sensitiveなマッチをする
    )


class DictMatchTyper(Typer):
    def __init__(self, conf: DictMatchTyperConfig) -> None:
        with open(os.path.join(get_original_cwd(), conf.term2cat), "rb") as f:
            self.term2cat = pickle.load(f)

        self.conf = conf
        # keyword extractorを追加する
        # argumentを追加する...後でいいか...
        self.keyword_processor = ComplexKeywordTyper(
            self.term2cat, case_sensitive=conf.case_sensitive
        )
        self.label_names = ["nc-O"] + list(set(self.term2cat.values()))

    def predict(
        self, tokens: List[str], starts: List[str], ends: List[str]
    ) -> TyperOutput:
        if not starts:
            return TyperOutput(labels=[], max_probs=np.array([]), probs=np.array([]))
        else:
            labels = []
            for start, end in zip(starts, ends):
                term = " ".join(tokens[start:end])
                label = self.keyword_processor.type_chunk(term)
                labels.append(label)
            label_ids = np.array([self.label_names.index(l) for l in labels])
            probs = np.eye(len(self.label_names))[label_ids]
            return TyperOutput(
                labels=labels, max_probs=np.ones(len(labels)), probs=probs
            )

    def train(self):
        pass
