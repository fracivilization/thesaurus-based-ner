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
from src.dataset.term2cat.term2cat import Term2CatConfig, load_term2cat
import numpy as np

np.ones(13)


@dataclass
class DictMatchTyperConfig(TyperConfig):
    typer_name: str = "DictMatchTyper"
    term2cat: Term2CatConfig = Term2CatConfig()
    label_names: str = "non_initialized"  # this variable is dinamically decided


class DictMatchTyper(Typer):
    def __init__(self, conf: DictMatchTyperConfig) -> None:
        self.term2cat = load_term2cat(conf.term2cat)
        self.conf = conf
        # keyword extractorを追加する
        # argumentを追加する...後でいいか...
        self.keyword_processor = ComplexKeywordTyper(self.term2cat)
        self.label_names = ["O"] + list(set(self.term2cat.values()))

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
