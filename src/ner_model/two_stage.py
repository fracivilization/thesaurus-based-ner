from src.ner_model.typer.abstract_model import TyperConfig
from typing import List
from hydra.core.config_store import ConfigStore
from .abstract_model import NERModelConfig, NERModel
from dataclasses import dataclass
from src.ner_model.chunker.abstract_model import Chunker, ChunkerConfig
from src.ner_model.typer.abstract_model import Typer
from datasets import DatasetDict
from omegaconf import MISSING


@dataclass
class TwoStageConfig(NERModelConfig):
    ner_model_name: str = "TwoStage"
    chunker: ChunkerConfig = MISSING
    typer: TyperConfig = MISSING
    pass


def register_chunker_configs() -> None:
    cs = ConfigStore.instance()
    from .chunker.flair_model import FlairNPChunkerConfig

    from .chunker.spacy_model import BeneparNPChunkerConfig

    cs.store(
        group="ner_model/chunker",
        name="FlairNPChunker",
        node=FlairNPChunkerConfig,
    )

    cs.store(
        group="ner_model/chunker",
        name="BeneparNPChunker",
        node=BeneparNPChunkerConfig,
    )


def register_typer_configs() -> None:
    cs = ConfigStore.instance()
    from .typer.dict_match_typer import DictMatchTyperConfig
    from .typer.inscon_typer import InsconTyperConfig

    cs.store(
        group="ner_model/typer",
        name="base_DictMatchTyper_config",
        node=DictMatchTyperConfig,
    )

    cs.store(
        group="ner_model/typer",
        name="base_InsconTyper_config",
        node=InsconTyperConfig,
    )


def register_two_stage_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="ner_model", name="base_TwoStage_model_config", node=TwoStageConfig)
    register_chunker_configs()
    register_typer_configs()


class TwoStageModel(NERModel):
    def __init__(self, chunker: Chunker, typer: Typer, datasets: DatasetDict) -> None:
        super().__init__()
        self.chunker = chunker
        self.typer = typer
        self.datasets = datasets

    def batch_predict(self, tokens: List[List[str]]) -> List[List[str]]:
        chunks = self.chunker.batch_predict(tokens)
        return super().batch_predict(tokens)

    def train(self):
        self.chunker.train()
        self.typer.train()
