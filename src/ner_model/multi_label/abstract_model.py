from omegaconf.omegaconf import MISSING
from typing import List, Tuple
from .ml_typer.abstract import MultiLabelTyperOutput
from dataclasses import dataclass


class MultiLabelNERModelConfig:
    multi_label_ner_model_name: str = MISSING


class MultiLabelNERModel:
    def __init__(self, conf: MultiLabelNERModelConfig, label_names: List[str]) -> None:
        self.conf: MultiLabelNERModelConfig = conf
        self.label_names = label_names

    def predict(
        self, tokens: List[str]
    ) -> Tuple[List[int], List[int], List[MultiLabelTyperOutput]]:
        starts: List[int] = None
        ends: List[int] = None
        labels: List[MultiLabelTyperOutput] = None
        raise NotImplementedError
        return starts, ends, labels

    def batch_predict(
        self, tokens: List[List[str]]
    ) -> Tuple[List[List[int]], List[List[int]], List[List[MultiLabelTyperOutput]]]:
        starts: List[List[int]] = None
        ends: List[List[int]] = None
        labels: List[List[MultiLabelTyperOutput]] = None
        raise NotImplementedError
        return starts, ends, labels
