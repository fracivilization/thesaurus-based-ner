from src.ner_model.chunker import chunker_builder
from src.ner_model.typer import typer_builder
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

    from .chunker.spacy_model import BeneparNPChunkerConfig, SpacyNPChunkerConfig

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

    cs.store(
        group="ner_model/chunker",
        name="SpacyNPChunker",
        node=SpacyNPChunkerConfig,
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
    def __init__(self, config: TwoStageConfig, datasets: DatasetDict) -> None:
        super().__init__()
        self.conf = config
        self.chunker = chunker_builder(config.chunker)
        self.typer = typer_builder(config.typer, datasets)
        self.datasets = datasets

    def predict(self, tokens: List[str]) -> List[str]:
        chunk = self.chunker.predict(tokens)
        starts = [s for s, e in chunk]
        ends = [e for s, e in chunk]
        types = self.typer.predict(tokens, starts, ends)
        ner_tags = ["O"] * len(tokens)
        assert len(starts) == len(ends)
        assert len(ends) == len(types)
        for s, e, l in zip(starts, ends, types):
            if l != "O":
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
        ner_tags = []
        for snt_tokens, snt_starts, snt_ends, snt_types in zip(
            tokens, starts, ends, types
        ):
            snt_ner_tags = ["O"] * len(snt_tokens)
            assert len(snt_starts) == len(snt_ends)
            assert len(snt_ends) == len(snt_types)
            for s, e, l in zip(snt_starts, snt_ends, snt_types):
                for i in range(s, e):
                    if i == s:
                        snt_ner_tags[i] = "B-%s" % l
                    else:
                        snt_ner_tags[i] = "I-%s" % l
            assert len(snt_tokens) == len(snt_ner_tags)
            ner_tags.append(snt_ner_tags)
        return ner_tags

    def train(self):
        self.chunker.train()
        self.typer.train()
