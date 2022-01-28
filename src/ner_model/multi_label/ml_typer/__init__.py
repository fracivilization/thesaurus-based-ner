from datasets import DatasetDict
from src.ner_model.chunker.abstract_model import Chunker

from src.utils.mlflow import MlflowWriter
from .abstract import MultiLabelTyperConfig
from .dict_match import MultiLabelDictMatchTyper
from .enumerated import MultiLabelEnumeratedTyper, MultiLabelEnumeratedTyperConfig
from seqeval.metrics.sequence_labeling import get_entities
from hydra.core.config_store import ConfigStore


def register_multi_label_typer_configs(group="multi_label_typer") -> None:
    cs = ConfigStore.instance()
    from .dict_match import MultiLabelDictMatchTyperConfig

    cs.store(
        group=group,
        name="MultiLabelDictMatchTyper",
        node=MultiLabelDictMatchTyperConfig,
    )
    cs.store(
        group=group,
        name="MultiLabelEnumeratedTyper",
        node=MultiLabelEnumeratedTyperConfig,
    )


def multi_label_typer_builder(
    config: MultiLabelTyperConfig,
    # datasets: DatasetDict,
    writer: MlflowWriter,
):
    # if datasets:
    #     label_names = sorted(
    #         set(datasets["train"].features["labels"].feature.feature.names)
    #     )
    #     config.label_names = repr(label_names)
    if config.multi_label_typer_name == "MultiLabelDictMatchTyper":
        return MultiLabelDictMatchTyper(config)
    elif config.multi_label_typer_name == "MultiLabelEnumeratedTyper":
        return MultiLabelEnumeratedTyper(config)
    else:
        raise NotImplementedError
