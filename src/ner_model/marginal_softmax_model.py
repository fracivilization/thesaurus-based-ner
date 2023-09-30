import uuid
from .abstract_model import NERModelConfig, NERModel
from dataclasses import dataclass
from .multi_label.abstract_model import MultiLabelNERModelConfig
from omegaconf.omegaconf import MISSING
from typing import List, Tuple, Optional
import numpy as np
from .multi_label.abstract_model import MultiLabelNERModel
from scipy.special import softmax
from hydra.core.config_store import ConfigStore
from .multi_label import register_multi_label_ner_model, multi_label_ner_model_builder
from datasets import Dataset
from src.dataset.utils import load_negative_cats_from_positive_cats
import multiprocessing
from p_tqdm import p_map
from more_itertools import chunked
from collections import defaultdict
from src.utils.typer_to_bio import call_postprocess_for_multi_label_ner_output
import copy


@dataclass
class FlattenMarginalSoftmaxNERModelConfig(NERModelConfig):
    ner_model_name: str = "FlattenMarginalSoftmaxNER"
    multi_label_ner_model: MultiLabelNERModelConfig = MISSING
    positive_cats: str = MISSING
    eval_dataset: str = MISSING  # CoNLL2003 or MedMentions
    with_negative_categories: bool = True


def batch_postprocess_for_output(examples):
    ner_tags = []
    for example in examples:
        ner_tags.append(call_postprocess_for_multi_label_ner_output(example))
    return ner_tags


class FlattenMarginalSoftmaxNERModel(NERModel):
    """Abstract Class for Evaluation"""

    def __init__(self, conf: FlattenMarginalSoftmaxNERModelConfig) -> None:
        self.conf: FlattenMarginalSoftmaxNERModelConfig = conf
        self.multi_label_ner_model: MultiLabelNERModel = multi_label_ner_model_builder(
            conf.multi_label_ner_model
        )  # 後で追加する
        label_names = self.multi_label_ner_model.multi_label_typer.label_names
        self.positive_cats = [
            cat for cat in self.conf.positive_cats.split("_") if cat in label_names
        ]
        if self.conf.with_negative_categories:
            self.negative_cats = []
            for cat in sorted(
                load_negative_cats_from_positive_cats(
                    self.positive_cats, conf.eval_dataset
                )
            ):
                if cat in label_names:
                    self.negative_cats.append(cat)
        else:
            self.negative_cats = []

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
        ml_label_names = self.multi_label_ner_model.label_names
        ner_tags = []
        examples = [
            {
                "tokens": snt_tokens,
                "starts": snt_starts,
                "ends": snt_ends,
                "outputs": snt_outputs,
                "ml_label_names": ml_label_names,
                "positive_cats": self.positive_cats,
                "negative_cats": self.negative_cats,
            }
            for snt_tokens, snt_starts, snt_ends, snt_outputs in zip(
                tokens, starts, ends, outputs
            )
        ]
        chunked_examples = list(
            chunked(examples, n=1 + len(examples) // (3 * multiprocessing.cpu_count()))
        )
        ner_tags = p_map(batch_postprocess_for_output, chunked_examples)
        ner_tags = [snt_ner_tags for chunk in ner_tags for snt_ner_tags in chunk]
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
