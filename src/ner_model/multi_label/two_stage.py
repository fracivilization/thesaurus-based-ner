from .ml_typer import (
    multi_label_typer_builder,
    register_multi_label_typer_configs,
)
from .abstract_model import MultiLabelNERModel, MultiLabelNERModelConfig
from omegaconf import MISSING
from ..chunker import ChunkerConfig, chunker_builder, register_chunker_configs
from dataclasses import dataclass
from .ml_typer.abstract import MultiLabelTyperConfig, MultiLabelTyperOutput
from hydra.core.config_store import ConfigStore
from datasets.dataset_dict import DatasetDict
from src.utils.mlflow import MlflowWriter
import numpy as np
from typing import Tuple, List


@dataclass
class MultiLabelTwoStageConfig(MultiLabelNERModelConfig):
    multi_label_ner_model_name: str = "MultiLabelTwoStage"
    chunker: ChunkerConfig = MISSING
    multi_label_typer: MultiLabelTyperConfig = MISSING


def register_multi_label_two_stage_configs(group="multi_label_ner_model") -> None:
    cs = ConfigStore.instance()
    cs.store(
        group=group,
        name="base_MultiLabelTwoStage_model_config",
        node=MultiLabelTwoStageConfig,
    )
    register_chunker_configs(group="%s/chunker" % group)
    register_multi_label_typer_configs(group="%s/multi_label_typer" % group)


# TODO: MultiLabelNERModelを別離したほうがいいかもしれないけどとりあえず...
class MultiLabelTwoStageModel(MultiLabelNERModel):
    def __init__(
        self,
        config: MultiLabelTwoStageConfig,
        writer: MlflowWriter,
    ) -> None:
        self.conf = config
        self.writer = writer
        self.chunker = chunker_builder(config.chunker)
        self.multi_label_typer = multi_label_typer_builder(
            config.multi_label_typer, writer
        )
        super().__init__(config, label_names=self.multi_label_typer.label_names)
        # self.datasets = datasets

    def predict(self, tokens: List[str]) -> List[str]:
        raise NotImplementedError

    def remove_null_chunk(
        self,
        input_starts: List[List[str]],
        input_ends: List[List[str]],
        input_outputs: List[List[MultiLabelTyperOutput]],
    ) -> Tuple[List[List[str]], List[List[str]], List[List[MultiLabelTyperOutput]]]:
        starts = []
        ends = []
        outputs = []
        for snt_input_starts, snt_input_ends, snt_input_outputs in zip(
            input_starts, input_ends, input_outputs
        ):
            snt_starts = []
            snt_ends = []
            snt_outputs = []
            for s, e, o in zip(
                snt_input_starts,
                snt_input_ends,
                snt_input_outputs
            ):
                if o.labels:
                    snt_starts.append(s)
                    snt_ends.append(e)
                    snt_outputs.append(o)
            starts.append(snt_starts)
            ends.append(snt_ends)
            outputs.append(snt_outputs)
        return starts, ends, outputs

    def batch_predict(
        self, tokens: List[List[str]]
    ) -> Tuple[List[List[int]], List[List[int]], List[List[MultiLabelTyperOutput]]]:
        chunks = self.chunker.batch_predict(tokens)
        starts = [[s for s, e in snt] for snt in chunks]
        ends = [[e for s, e in snt] for snt in chunks]
        outputs = self.multi_label_typer.batch_predict(tokens, starts, ends)
        starts, ends, outputs = self.remove_null_chunk(starts, ends, outputs)
        return starts, ends, outputs

    def train(self):
        self.chunker.train()
        self.multi_label_typer.train()
