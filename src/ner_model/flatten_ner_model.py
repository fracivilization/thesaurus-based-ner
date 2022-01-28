from matplotlib.pyplot import axis
from .abstract_model import NERModelConfig, NERModel
from dataclasses import dataclass
from .multi_label.abstract_model import MultiLabelNERModelConfig
from .multi_label.ml_typer.abstract import MultiLabelTyperOutput
from omegaconf.omegaconf import MISSING
from typing import List, Tuple
import numpy as np
from .multi_label.abstract_model import MultiLabelNERModel
from scipy.special import softmax


@dataclass
class FlattenMultiLabelNERModelConfig(NERModelConfig):
    ner_model_name: str = "FlattenMultiLabelNER"
    multi_label_ner_model: MultiLabelNERModelConfig = MISSING
    focus_cats: str = MISSING


class FlattenMultiLabelNERModel(NERModel):
    """Abstract Class for Evaluation"""

    def __init__(self, conf: FlattenMultiLabelNERModelConfig) -> None:
        self.conf: FlattenMultiLabelNERModel = NERModelConfig()
        self.multi_label_ner_model: MultiLabelNERModel = None  # 後で追加する
        self.focus_cats = ["nc-O"] + self.conf.focus_cats.split("_")
        self.focus_label_ids = np.array(
            [
                label_id
                for label_id, label in enumerate(self.multi_label_typer.label_names)
                if label == "nc-O" or label in self.focus_cats
            ]
        )

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
        starts, ends, labels = self.multi_label_ner_model.batch_predict(tokens)
        ner_tags = []
        for snt_tokens, snt_starts, snt_ends, snt_type_and_max_probs in zip(
            tokens, starts, ends, labels
        ):
            logits = snt_type_and_max_probs.logits  # (span_num, label_num)
            focus_logits = np.take_along_axis(
                logits, np.tile(self.focus_label_ids[None, :], (len(logits), 1)), axis=1
            )  # (span_num, focus_cat_num)
            snt_types = [
                self.focus_cats[max_id] for max_id in focus_logits.argmax(axis=1)
            ]
            max_probs = softmax(focus_logits, axis=1).max(axis=1)
            snt_ner_tags = ["O"] * len(snt_tokens)
            assert len(snt_starts) == len(snt_ends)
            assert len(snt_ends) == len(snt_types)
            labeled_chunks = sorted(
                zip(snt_starts, snt_ends, snt_types, max_probs),
                key=lambda x: x[3],
                reverse=True,
            )
            for s, e, label, max_prob in labeled_chunks:
                if not label.startswith("nc-") and all(
                    tag == "O" for tag in snt_ner_tags[s:e]
                ):
                    for i in range(s, e):
                        if i == s:
                            snt_ner_tags[i] = "B-%s" % label
                        else:
                            snt_ner_tags[i] = "I-%s" % label
            assert len(snt_tokens) == len(snt_ner_tags)
            ner_tags.append(snt_ner_tags)
        return ner_tags

    def train(self):
        raise NotImplementedError
