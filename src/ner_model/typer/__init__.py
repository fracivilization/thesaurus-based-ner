from datasets import DatasetDict

from src.utils.mlflow import MlflowWriter
from .abstract_model import SpanClassifierNERWrapper, TyperConfig
from .dict_match_typer import DictMatchTyper
from .inscon_typer import InsconTyper
from .enumerated_typer import EnumeratedTyper


def typer_builder(config: TyperConfig, ner_datasets: DatasetDict, writer: MlflowWriter):
    if config.typer_name == "DictMatchTyper":
        # raise NotImplementedError
        return DictMatchTyper(config)
    elif config.typer_name == "Inscon":
        return InsconTyper(config, ner_datasets)
    elif config.typer_name == "Enumerated":
        return EnumeratedTyper(config, ner_datasets)
