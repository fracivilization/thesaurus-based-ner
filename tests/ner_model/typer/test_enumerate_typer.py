import unittest
from src.ner_model.typer.enumerated_typer import EnumeratedTyper, EnumeratedTyperConfig
from datasets import DatasetDict
from src.utils.hydra import HydraAddaptedTrainingArguments

MSC_DATASET_PATH = "tests/fixtures/mini_conll_msc_dataset"


class TestEnumeratedTyper(unittest.TestCase):
    def test_load_enumerated_typer(self):
        config = EnumeratedTyperConfig(msc_datasets=MSC_DATASET_PATH)
        EnumeratedTyper(config)

    def test_predict_enumerated_typer(self):
        msc_dataset = DatasetDict.load_from_disk(MSC_DATASET_PATH)
        config = EnumeratedTyperConfig(msc_datasets=MSC_DATASET_PATH)
        typer = EnumeratedTyper(config)
        validation_dataset = msc_dataset["validation"]
        tokens, starts, ends = (
            validation_dataset["tokens"],
            validation_dataset["starts"],
            validation_dataset["ends"],
        )
        typer.batch_predict(tokens, starts, ends)

    def test_train_euumerated_typer(self):
        config = EnumeratedTyperConfig(msc_datasets=MSC_DATASET_PATH)
        typer = EnumeratedTyper(config)
        typer.train()

    def test_train_euumerated_typer_with_16bit(self):
        # train_args で 16bitを指定
        train_args = HydraAddaptedTrainingArguments(fp16=True, output_dir=".")
        config = EnumeratedTyperConfig(
            msc_datasets=MSC_DATASET_PATH, train_args=train_args
        )
        typer = EnumeratedTyper(config)
        typer.train()
