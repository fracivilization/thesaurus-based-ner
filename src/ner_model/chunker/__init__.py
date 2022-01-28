from .abstract_model import ChunkerConfig, EnumeratedChunker
from .flair_model import FlairNPChunker
from .spacy_model import BeneparNPChunker, SpacyNPChunker
from hydra.core.config_store import ConfigStore


def chunker_builder(config: ChunkerConfig):
    if config.chunker_name == "FlairNPChunker":
        return FlairNPChunker()
    elif config.chunker_name == "BeneparNPChunker":
        return BeneparNPChunker()
    elif config.chunker_name == "SpacyNPChunker":
        return SpacyNPChunker(config)
    elif config.chunker_name == "enumerated":
        return EnumeratedChunker(config)
    else:
        raise NotImplementedError


def register_chunker_configs(group="ner_model/chunker") -> None:
    cs = ConfigStore.instance()
    from .flair_model import FlairNPChunkerConfig
    from .spacy_model import BeneparNPChunkerConfig, SpacyNPChunkerConfig
    from .abstract_model import EnumeratedChunkerConfig

    cs.store(
        group=group,
        name="FlairNPChunker",
        node=FlairNPChunkerConfig,
    )

    cs.store(
        group=group,
        name="BeneparNPChunker",
        node=BeneparNPChunkerConfig,
    )

    cs.store(
        group=group,
        name="SpacyNPChunker",
        node=SpacyNPChunkerConfig,
    )
    cs.store(
        group=group,
        name="EnumeratedChunker",
        node=EnumeratedChunkerConfig,
    )
