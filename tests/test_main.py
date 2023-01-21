from pandas.core.indexing import convert_to_index_sliceable
from logging import getLogger

logger = getLogger(__name__)


def test_main():
    pass


def test_spacy_chunker():
    from src.ner_model.chunker.spacy_model import SpacyNPChunkerConfig, SpacyNPChunker

    chunker = SpacyNPChunker(SpacyNPChunkerConfig())
    chunker.predict(["I", "want", "to", "be", "Pokemon", "Master", "."])
