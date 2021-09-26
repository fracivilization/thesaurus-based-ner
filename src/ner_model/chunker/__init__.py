from .abstract_model import ChunkerConfig
from .flair_model import FlairNPChunker
from .spacy_model import BeneparNPChunker, SpacyNPChunker


def chunker_builder(config: ChunkerConfig):
    if config.chunker_name == "FlairNPChunker":
        return FlairNPChunker()
    elif config.chunker_name == "BeneparNPChunker":
        return BeneparNPChunker()
    elif config.chunker_name == "SpacyNPChunker":
        return SpacyNPChunker(config)
    else:
        raise NotImplementedError
