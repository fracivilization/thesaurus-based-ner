from typing import List
from hydra.core.config_store import ConfigStore
from .abstract_model import NERModelConfig, NERModel
from dataclasses import dataclass
from ner_model.chunker.abstract_model import Chunker
from ner_model.typer.abstract_model import Typer
from datasets import DatasetDict


@dataclass
class TwoStageConfig(NERModelConfig):
    ner_model_name: str = "TwoStage"
    chunker: str = "SpacyNP"
    typer: str = "Inscon"
    pass


def register_two_stage_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="ner_model", name="base_TwoStage_model_config", node=TwoStageConfig)


class TwoStageModel(NERModel):
    def __init__(self, chunker: Chunker, typer: Typer, datasets: DatasetDict) -> None:
        super().__init__()
        self.chunker = chunker
        self.typer = typer
        self.datasets = datasets

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return super().batch_predict(tokens)

    def train(self):
        self.chunker.train()
        self.typer.train()
