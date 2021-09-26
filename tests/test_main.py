def test_main():
    pass


def test_random_chunker():
    from src.chunking.abstract_model import RandomChunker

    chunker = RandomChunker()
    chunks = chunker.predict("I have pens and what kinds of pens do you want ?".split())
    for s, e in chunks:
        assert s < e


def test_spacy_chunker():
    from src.ner_model.chunker.spacy_model import SpacyNPChunkerConfig, SpacyNPChunker

    chunker = SpacyNPChunker(SpacyNPChunkerConfig())
    chunker.predict(["I", "want", "to", "be", "Pokemon", "Master", "."])
