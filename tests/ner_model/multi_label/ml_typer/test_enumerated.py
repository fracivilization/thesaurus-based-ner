import unittest
from src.ner_model.multi_label.ml_typer import (
    MultiLabelEnumeratedTyper,
    MultiLabelEnumeratedTyperConfig,
)
from datasets import DatasetDict

MSMLC_DATASET_PATH = "tests/fixtures/mini_conll_msmlc_dataset"


class TestEnumeratedTyper(unittest.TestCase):
    def test_load_enumerated_typer(self):
        config = MultiLabelEnumeratedTyperConfig(train_datasets=MSMLC_DATASET_PATH)
        MultiLabelEnumeratedTyper(config)

    def test_predict_enumerated_typer(self):
        msmlc_dataset = DatasetDict.load_from_disk(MSMLC_DATASET_PATH)
        config = MultiLabelEnumeratedTyperConfig(train_datasets=MSMLC_DATASET_PATH)
        ml_typer = MultiLabelEnumeratedTyper(config)
        validation_dataset = msmlc_dataset["validation"]
        tokens, starts, ends = (
            validation_dataset["tokens"],
            validation_dataset["starts"],
            validation_dataset["ends"],
        )
        ml_typer.batch_predict(tokens, starts, ends)

    def test_train_enumerated_typer(self):
        config = MultiLabelEnumeratedTyperConfig(train_datasets=MSMLC_DATASET_PATH)
        ml_typer = MultiLabelEnumeratedTyper(config)
        ml_typer.train()
