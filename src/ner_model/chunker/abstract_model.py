from collections import defaultdict
from datasets import DatasetDict, Dataset, Sequence, Value
from typing import Dict, List, Tuple
from datasets.info import DatasetInfo
from dataclasses import MISSING, dataclass, field
import numpy as np
from tqdm import tqdm
from omegaconf import MISSING

Span = Tuple[int, int]


@dataclass
class ChunkerConfig:
    chunker_name: str = MISSING


class Chunker:
    def __init__(
        self,
        span_detection_datasets: DatasetDict,
    ) -> None:
        self.span_detection_datasets = span_detection_datasets
        self.argss = []

    def predict(self, tokens: List[str]) -> List[Span]:
        raise NotImplementedError

    def batch_predict(
        self,
        tokens: List[List[str]],
    ) -> List[Span]:
        return [self.predict(token) for token in tokens]

    def preprocess(self):
        self.span_detection_datasets = self.span_detection_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,  # because it might have a problem
        )

    def preprocess_function(self, example: Dict) -> Dict:
        """preprocess_function for encoding
        Args:
            example (Dict): {"tokens": List[List[str]], "start": List[int], "end": List[int], "label": List[int]}
        Returns:
            ret_dict (Dict): {"label": List[int], "...": ...}
        """
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
