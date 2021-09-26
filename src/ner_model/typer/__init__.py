from datasets import DatasetDict
from .abstract_model import SpanClassifierNERWrapper, TyperConfig
from .dict_match_typer import DictMatchTyper
from .inscon_typer import InsconTyper


def typer_builder(config: TyperConfig, datasets: DatasetDict):
    if config.typer_name == "DictMatchTyper":
        # raise NotImplementedError
        return DictMatchTyper(config)
    elif config.typer_name == "Inscon":
        return InsconTyper(config, datasets)
    pass
