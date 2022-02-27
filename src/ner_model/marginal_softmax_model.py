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
from hydra.core.config_store import ConfigStore
from .multi_label import register_multi_label_ner_model, multi_label_ner_model_builder


@dataclass
class FlattenMarginalSoftmaxNERModelConfig(NERModelConfig):
    ner_model_name: str = "FlattenMarginalSoftmaxNER"
    multi_label_ner_model: MultiLabelNERModelConfig = MISSING
    focus_cats: str = MISSING


class FlattenMarginalSoftmaxNERModel(NERModel):
    """Abstract Class for Evaluation"""

    def __init__(self, conf: FlattenMarginalSoftmaxNERModelConfig) -> None:
        self.conf: FlattenMarginalSoftmaxNERModelConfig = conf
        self.multi_label_ner_model: MultiLabelNERModel = multi_label_ner_model_builder(
            conf.multi_label_ner_model
        )  # 後で追加する
        self.focus_cats = ["nc-O"] + self.conf.focus_cats.split("_")
        self.focus_label_ids = np.array(
            [
                label_id
                for label_id, label in enumerate(
                    self.multi_label_ner_model.multi_label_typer.label_names
                )
                if label in self.focus_cats
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
        starts, ends, outputs = self.multi_label_ner_model.batch_predict(tokens)
        label_names = self.multi_label_ner_model.label_names
        ner_tags = []
        focus_cats = set(self.focus_cats)
        for snt_tokens, snt_starts, snt_ends, snt_outputs in zip(
            tokens, starts, ends, outputs
        ):
            remained_starts = []
            remained_ends = []
            remained_labels = []
            max_logits = []
            for s, e, o in zip(snt_starts, snt_ends, snt_outputs):
                # focus_catsが何かしら出力されているスパンのみ残す
                # 更に残っている場合は最大確率のものを残す
                focus_logits = o.logits[self.focus_label_ids]
                max_logit = focus_logits.max()
                max_logit = max(focus_logits)
                remained_starts.append(s)
                remained_ends.append(e)
                remained_labels.append(self.focus_cats[focus_logits.argmax()])
                max_logits.append(max_logit)
            labeled_chunks = sorted(
                zip(remained_starts, remained_ends, remained_labels, max_logits),
                key=lambda x: x[3],
                reverse=True,
            )
            snt_ner_tags = ["O"] * len(snt_tokens)
            for s, e, label, max_logit in labeled_chunks:
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
        self.multi_label_ner_model.train()


def register_flattern_marginal_softmax_ner_configs(group="ner_model"):
    cs = ConfigStore()
    cs.store(
        group=group,
        name="FlattenMarginalSoftmaxNER",
        node=FlattenMarginalSoftmaxNERModelConfig,
    )
    register_multi_label_ner_model("%s/multi_label_ner_model" % group)
    # raise NotImplementedError
