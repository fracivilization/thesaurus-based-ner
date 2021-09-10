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


@dataclass
class DictMatchTyperConfig(TyperConfig):
    typer_name: str = "DictMatchTyper"


class DictMatchTyper(Typer):
    def __init__(self, term2cat: Dict) -> None:
        self.term2cat = term2cat
        self.argss = ["DictMatchClassifier", self.term2cat]
        # keyword extractorを追加する
        # argumentを追加する...後でいいか...
        self.keyword_processor = ComplexKeywordProcessor(self.term2cat)

    def predict(self, tokens: List[str], start: int, end: int) -> SpanClassifierOutput:
        term = " ".join(tokens[start:end])
        keywords = self.keyword_processor.extract_keywords(term)
        end_match_keyword = [(l, s, e) for l, s, e in keywords if e == len(term)]
        if end_match_keyword:
            l, s, e = sorted(end_match_keyword, key=lambda x: x[1])[0]
            return SpanClassifierOutput(label=l)
        else:
            return SpanClassifierOutput(label="O")

    def batch_predict(
        self, tokens: List[List[str]], start: List[int], end: List[int]
    ) -> List[SpanClassifierOutput]:
        return [self.predict(tok, s, e) for tok, s, e in tqdm(zip(tokens, start, end))]
