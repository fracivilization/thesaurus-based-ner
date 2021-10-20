from typing import List
from dataclasses import dataclass

from omegaconf.omegaconf import MISSING


@dataclass
class NERModelWrapperConfig:
    o_label_as_Bo: bool = MISSING


@dataclass
class NERModelConfig:
    ner_model_name: str = MISSING
    ner_model_wrapper: NERModelWrapperConfig = NERModelWrapperConfig(
        o_label_as_Bo=False
    )


class NERModel:
    """Abstract Class for Evaluation"""

    def __init__(self) -> None:
        self.conf: NERModelConfig = NERModelConfig()

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


class NERModelWrapper(NERModel):
    def __init__(self, ner_model: NERModel, config: NERModelConfig) -> None:
        self.ner_model = ner_model
        self.conf = config

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        predicted_tags = self.ner_model.batch_predict(tokens)
        if self.conf.ner_model_wrapper.o_label_as_Bo:
            predicted_tags = [self.wrap_predicted_tags(tags) for tags in predicted_tags]
        return predicted_tags

    def predict(self, tokens: List[str]) -> List[str]:
        predicted_tags = self.ner_model.predict(tokens)
        if self.conf.ner_model_wrapper.o_label_as_Bo:
            predicted_tags = self.wrap_predicted_tags(predicted_tags)
        return predicted_tags

    def wrap_predicted_tags(self, tags: List[str]) -> List[str]:
        new_tags = tags
        for i, tag in enumerate(tags):
            if tag == "O":
                new_tags[i] = "B-nc-O"
        return new_tags

    def train(self):
        pass
