from typing import Dict, List
from .abstract_model import (
    Typer,
    SpanClassifierDataTrainingArguments,
    SpanClassifierOutput,
    TyperConfig,
)
from datasets import DatasetDict
from src.ner_model.matcher_model import ComplexKeywordProcessor
from tqdm import tqdm
from dataclasses import dataclass
from omegaconf import MISSING
from src.dataset.term2cat.term2cat import Term2CatConfig, load_term2cat


@dataclass
class DictMatchTyperConfig(TyperConfig):
    typer_name: str = "DictMatchTyper"
    term2cat: Term2CatConfig = Term2CatConfig()


class DictMatchTyper(Typer):
    def __init__(self, conf: DictMatchTyperConfig) -> None:
        self.term2cat = load_term2cat(conf.term2cat)
        self.argss = conf
        # keyword extractorを追加する
        # argumentを追加する...後でいいか...
        self.keyword_processor = ComplexKeywordProcessor(self.term2cat)

    def predict(
        self, tokens: List[str], starts: List[str], ends: List[str]
    ) -> List[str]:
        labels = []
        for start, end in zip(starts, ends):
            term = " ".join(tokens[start:end])
            keywords = self.keyword_processor.extract_keywords(term)
            end_match_keyword = [(l, s, e) for l, s, e in keywords if e == len(term)]
            if end_match_keyword:
                l, s, e = sorted(end_match_keyword, key=lambda x: x[1])[0]
                labels.append(l)
            else:
                labels.append("O")
        return labels
