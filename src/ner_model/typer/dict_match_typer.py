from typing import Dict, List, Type
from .abstract_model import (
    Typer,
    SpanClassifierDataTrainingArguments,
    SpanClassifierOutput,
    TyperConfig,
    TyperOutput,
)
from datasets import DatasetDict
from src.ner_model.matcher_model import ComplexKeywordTyper
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
        self.label_names = eval(self.conf.label_names)

    def predict(
        self, tokens: List[str], starts: List[str], ends: List[str]
    ) -> TyperOutput:
        labels = []
        for start, end in zip(starts, ends):
            term = " ".join(tokens[start:end])
            label = self.keyword_processor.type_chunk(term)
            assert label != "O"
            labels.append(label)
        label_ids = np.array([self.label_names.index(l) for l in labels])
        np.eye(len(self.conf.label_names))[label_ids]
        return TyperOutput(labels=labels, max_probs=np.ones(len(labels)), probs=probs)

    def train(self):
        pass
