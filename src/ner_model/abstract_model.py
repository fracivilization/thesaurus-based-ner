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
class NERModelWrapperConfig:
    o_label_as_Bo: bool = MISSING


@dataclass
class NERModelConfig:
    ner_model_name: str = MISSING
    ner_model_wrapper: NERModelWrapperConfig = NERModelWrapperConfig(
        o_label_as_Bo=False
    )


class NERModelWrapper(NERModel):
    def __init__(self, ner_model: NERModel, config: NERModelConfig) -> None:
        self.ner_model = ner_model
        self.conf = config

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        predicted_tags = self.ner_model.batch_predict(tokens)
        return [self.wrap_predicted_tags(tags) for tags in predicted_tags]

    def predict(self, tokens: List[str]) -> List[str]:
        predicted_tags = self.ner_model.predict(tokens)
        return self.wrap_predicted_tags(predicted_tags)

    def wrap_predicted_tags(self, tags: List[str]) -> List[str]:
        new_tags = tags
        for i, tag in enumerate(tags):
            if tag == "O":
                new_tags[i] = "B-nc-O"
        return new_tags

    def train(self):
        pass
