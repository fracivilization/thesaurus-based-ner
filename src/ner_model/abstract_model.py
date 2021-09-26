from typing import List
from dataclasses import dataclass

from omegaconf.omegaconf import MISSING


class NERModel:
    """Abstract Class for Evaluation"""

    def __init__(self) -> None:
        self.conf = dict()
        self.label_names = []

    def __post_init__(self):
        # Please specify args and labels names
        assert self.conf != dict()
        assert self.label_names != []

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
        return [self.predict(tok) for tok in tokens]

    def train(self):
        raise NotImplementedError


@dataclass
class NERModelConfig:
    ner_model_name: str = MISSING
