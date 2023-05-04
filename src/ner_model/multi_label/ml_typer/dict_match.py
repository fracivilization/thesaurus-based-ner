from typing import Dict, List, Type
from .abstract import (
    MultiLabelTyper,
    MultiLabelTyperConfig,
    MultiLabelTyperOutput,
)
from src.utils.string_match import ComplexKeywordTyper
from tqdm import tqdm
from dataclasses import dataclass
from omegaconf import MISSING
import numpy as np
from hydra.utils import to_absolute_path
from src.utils.utils import WeightedSQliteDict

np.ones(13)


@dataclass
class MultiLabelDictMatchTyperConfig(MultiLabelTyperConfig):
    multi_label_typer_name: str = "MultiLabelDictMatchTyper"
    term2cats: str = MISSING  # path for picled term2cat
    label_names: str = "non_initialized"  # this variable is dinamically decided
    case_sensitive: bool = False


class MultiLabelDictMatchTyper(MultiLabelTyper):
    def __init__(self, conf: MultiLabelDictMatchTyperConfig) -> None:
        self.conf = conf
        # term2catを無理やり作る(複数カテゴリを連結した文字列をTypeにしてしまえばそれで事足りそう)
        self.term2cats = WeightedSQliteDict.load_from_disk(
            to_absolute_path(conf.term2cats)
        )
        self.label_names = sorted(
            set(
                cat
                for weighted_cats in tqdm(self.term2cats.values())
                for cat in weighted_cats.values
            )
        )
        self.keyword_processor = ComplexKeywordTyper(
            self.term2cats, case_sensitive=conf.case_sensitive
        )

    def predict(
        self, tokens: List[str], starts: List[str], ends: List[str]
    ) -> List[MultiLabelTyperOutput]:
        assert len(starts) == len(ends)
        if not starts:
            return []
        else:
            weighted_labels = []
            for start, end in zip(starts, ends):
                term = " ".join(tokens[start:end])
                label = self.keyword_processor.type_chunk(term)
                if label == "nc-O":
                    weighted_labels.append(list())
                else:
                    weighted_labels.append(label)
            # label_ids = np.array([self.label_names.index(l) for l in labels])
            outputs = []
            for span_weighted_labels in weighted_labels:
                if span_weighted_labels:
                    span_labels = span_weighted_labels.values
                    span_weights = np.array(span_weighted_labels.weights)
                    label_ids = np.array(
                        [self.label_names.index(label) for label in span_labels]
                    )
                    logit = (
                        span_weights[:, None] * np.eye(len(self.label_names))[label_ids]
                    ).sum(axis=0)
                else:
                    span_labels = []
                    span_weights = []
                    logit = np.zeros(len(self.label_names))

                outputs.append(
                    MultiLabelTyperOutput(
                        labels=span_labels,
                        weights=span_weights,
                        logits=logit
                    )
                )

            return outputs

    def train(self):
        pass
