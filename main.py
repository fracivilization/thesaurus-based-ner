def test_random_chunker():
    from src.chunking.abstract_model import RandomChunker

    chunker = RandomChunker()
    chunks = chunker.predict("I have pens and what kinds of pens do you want ?".split())
    for s, e in chunks:
        assert s < e
