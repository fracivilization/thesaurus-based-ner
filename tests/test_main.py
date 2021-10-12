from pandas.core.indexing import convert_to_index_sliceable


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


def test_negative_evaluate():
    from src.evaluator import (
        calculate_negative_token_PRF,
        calculate_negative_precision,
    )
    from datasets import Dataset

    prediction_w_fn = Dataset.load_from_disk(
        "outputs/2021-10-08/11-34-12/data/output/524e086576e9287b3a5ad17b88bfffef/prediction_for_test.dataset"
    )
    gold_ner_tags = prediction_w_fn["gold_ner_tags"]
    pred_ner_tags = prediction_w_fn["pred_ner_tags"]
    negative_precision = calculate_negative_precision(gold_ner_tags, pred_ner_tags)
    assert 0 <= negative_precision and negative_precision <= 1
    negative_token_recall = calculate_negative_token_PRF(gold_ner_tags, pred_ner_tags)
    assert 0 <= negative_token_recall and negative_token_recall <= 1
    print("negative precision* %.2f" % (100 * negative_precision,))
    print("negative token recall* %.2f" % (100 * negative_token_recall,))
