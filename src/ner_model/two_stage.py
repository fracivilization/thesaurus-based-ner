from src.ner_model.chunker import chunker_builder, register_chunker_configs
from src.ner_model.typer import register_typer_configs, typer_builder
from src.ner_model.typer.abstract_model import TyperConfig
from typing import List, Tuple
from hydra.core.config_store import ConfigStore

from src.utils.mlflow import MlflowWriter
from .abstract_model import NERModelConfig, NERModel
from dataclasses import dataclass
from src.ner_model.chunker.abstract_model import (
    Chunker,
    ChunkerConfig,
    EnumeratedChunker,
)
from src.ner_model.typer.abstract_model import Typer
from datasets import DatasetDict
from omegaconf import MISSING
import numpy as np
from scipy.special import softmax


@dataclass
class TwoStageConfig(NERModelConfig):
    ner_model_name: str = "TwoStage"
    chunker: ChunkerConfig = MISSING
    typer: TyperConfig = MISSING


def register_two_stage_configs(group="ner_model") -> None:
    cs = ConfigStore.instance()
    cs.store(group=group, name="base_TwoStage_model_config", node=TwoStageConfig)
    register_chunker_configs(group="%s/chunker" % group)
    register_typer_configs(group="%s/typer" % group)


class TwoStageModel(NERModel):
    def __init__(
        self, config: TwoStageConfig, datasets: DatasetDict, writer: MlflowWriter
    ) -> None:
        super().__init__()
        self.conf = config
        self.writer = writer
        self.chunker = chunker_builder(config.chunker)
        self.typer = typer_builder(config.typer, datasets, writer, self.chunker)
        self.datasets = datasets

    def predict(self, tokens: List[str]) -> List[str]:
        chunk = self.chunker.predict(tokens)
        starts = [s for s, e in chunk]
        ends = [e for s, e in chunk]
        typer_output = self.typer.predict(tokens, starts, ends)
        types = typer_output.labels
        ner_tags = ["O"] * len(tokens)
        assert len(starts) == len(ends)
        assert len(ends) == len(types)
        for s, e, l in zip(starts, ends, types):
            if l != "nc-O":
                for i in range(s, e):
                    if i == s:
                        ner_tags[i] = "B-%s" % l
                    else:
                        ner_tags[i] = "I-%s" % l

        assert len(tokens) == len(ner_tags)
        return ner_tags

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        chunks = self.chunker.batch_predict(tokens)
        starts = [[s for s, e in snt] for snt in chunks]
        ends = [[e for s, e in snt] for snt in chunks]
        types = self.typer.batch_predict(tokens, starts, ends)
        # chunksとtypesを組み合わせて、BIO形式に変換する

    def train(self):
        self.chunker.train()
        self.typer.train()
