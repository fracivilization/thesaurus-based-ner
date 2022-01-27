from datasets import DatasetDict
from src.utils.mlflow import MlflowWriter
from .abstract_model import TyperConfig, RandomTyper
from .dict_match_typer import DictMatchTyper
from .inscon_typer import InsconTyper
from .enumerated_typer import EnumeratedTyper
from seqeval.metrics.sequence_labeling import get_entities
from hydra.core.config_store import ConfigStore


def typer_builder(
    config: TyperConfig,
    ner_datasets: DatasetDict,
    writer: MlflowWriter,
):
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
        return EnumeratedTyper(config)
    elif config.typer_name == "Random":
        return RandomTyper(config, ner_datasets)


def register_typer_configs(group="ner_model/typer") -> None:
    cs = ConfigStore.instance()
    from .dict_match_typer import DictMatchTyperConfig
    from .inscon_typer import InsconTyperConfig
    from .enumerated_typer import EnumeratedTyperConfig
    from .abstract_model import RandomTyperConfig
    from src.dataset.term2cat.term2cat import register_term2cat_configs

    register_term2cat_configs()
    cs.store(
        group=group,
        name="base_DictMatchTyper_config",
        node=DictMatchTyperConfig,
    )

    cs.store(
        group=group,
        name="base_InsconTyper_config",
        node=InsconTyperConfig,
    )

    cs.store(
        group=group,
        name="base_EnumeratedTyper_config",
        node=EnumeratedTyperConfig,
    )

    cs.store(
        group=group,
        name="base_RandomTyper_config",
        node=RandomTyperConfig,
    )
