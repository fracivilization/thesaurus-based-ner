from datasets import DatasetDict

from src.utils.mlflow import MlflowWriter
from .abstract_model import SpanClassifierNERWrapper, TyperConfig, RandomTyper
from .dict_match_typer import DictMatchTyper
from .inscon_typer import InsconTyper
from .enumerated_typer import EnumeratedTyper
from seqeval.metrics.sequence_labeling import get_entities


def typer_builder(config: TyperConfig, ner_datasets: DatasetDict, writer: MlflowWriter):
    if ner_datasets:
        label_names = list(
            set(
                [
                    l
                    for l, s, e in get_entities(
                        ner_datasets["train"].features["ner_tags"].feature.names
                    )
                ]
            )
        )
        config.label_names = repr(label_names)
    if config.typer_name == "DictMatchTyper":
        return DictMatchTyper(config)
    elif config.typer_name == "Inscon":
        return InsconTyper(config, ner_datasets)
    elif config.typer_name == "Enumerated":
        return EnumeratedTyper(config, ner_datasets)
    elif config.typer_name == "Random":
        return RandomTyper(config, ner_datasets)
