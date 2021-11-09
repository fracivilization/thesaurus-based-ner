from .abstract_model import ChunkerConfig, EnumeratedChunker
from .flair_model import FlairNPChunker
from .spacy_model import BeneparNPChunker, SpacyNPChunker


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
