import unittest
from src.ner_model.typer.enumerated_typer import EnumeratedTyper, EnumeratedTyperConfig
from datasets import DatasetDict
from src.utils.hydra import HydraAddaptedTrainingArguments

MSC_DATASET_PATH = "tests/fixtures/mini_conll_msc_dataset"
NER_DATASET_PATH = "tests/fixtures/mini_conll_ner_dataset"


class TestEnumeratedTyper(unittest.TestCase):
    def test_load_enumerated_typer(self):
        config = EnumeratedTyperConfig(train_msc_datasets=MSC_DATASET_PATH)
        EnumeratedTyper(config)

    def test_predict_enumerated_typer(self):
        msc_dataset = DatasetDict.load_from_disk(MSC_DATASET_PATH)
        config = EnumeratedTyperConfig(train_msc_datasets=MSC_DATASET_PATH)
        typer = EnumeratedTyper(config)
        validation_dataset = msc_dataset["validation"]
        tokens, starts, ends = (
            validation_dataset["tokens"],
            validation_dataset["starts"],
            validation_dataset["ends"],
        )
        typer.batch_predict(tokens, starts, ends)

    def test_train_euumerated_typer(self):
        config = EnumeratedTyperConfig(train_msc_datasets=MSC_DATASET_PATH)
        typer = EnumeratedTyper(config)
        typer.train()

    def test_train_euumerated_typer_with_16bit(self):
        # train_args で 16bitを指定
        train_args = HydraAddaptedTrainingArguments(fp16=True, output_dir=".")
        config = EnumeratedTyperConfig(
            train_msc_datasets=MSC_DATASET_PATH, train_args=train_args
        )
        typer = EnumeratedTyper(config)
        typer.train()

    def test_train_early_stopping_on_enumerated_typer(self):
        # c.f. https://dev.classmethod.jp/articles/huggingface-usage-early-stopping/
        train_args = HydraAddaptedTrainingArguments(
            output_dir="tmp",
            load_best_model_at_end=True,
            num_train_epochs=20,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="f1",
            greater_is_better=True,
            do_train=True,
            do_eval=True,
            overwrite_output_dir=True
        )
        config = EnumeratedTyperConfig(
            train_msc_datasets=MSC_DATASET_PATH,
            validation_ner_datasets=NER_DATASET_PATH,
            train_args=train_args
        )
        typer = EnumeratedTyper(config)
        typer.train()
