import unittest
from src.ner_model.multi_label.ml_typer import (
    MultiLabelEnumeratedTyper,
    MultiLabelEnumeratedTyperConfig,
)
from src.ner_model.multi_label.ml_typer.enumerated import MultiLabelEnumeratedModelArguments
from datasets import DatasetDict
from src.utils.hydra import HydraAddaptedTrainingArguments

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

    def test_train_euumerated_typer_with_16bit(self):
        # train_args で 16bitを指定
        train_args = HydraAddaptedTrainingArguments(fp16=True, output_dir=".")
        config = MultiLabelEnumeratedTyperConfig(
            train_datasets=MSMLC_DATASET_PATH, train_args=train_args
        )
        typer = MultiLabelEnumeratedTyper(config)
        typer.train()

    def test_train_euumerated_typer_with_early_stopping(self):
        # c.f. https://dev.classmethod.jp/articles/huggingface-usage-early-stopping/
        train_args = HydraAddaptedTrainingArguments(
            output_dir="tmp",
            load_best_model_at_end=True,
            num_train_epochs=20,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            do_train=True,
            overwrite_output_dir=True
        )
        model_args = MultiLabelEnumeratedModelArguments(
            model_name_or_path='bert-base-cased',
            loss_func="MarginalCrossEntropyLoss",
            dynamic_pn_ratio_equivalence=False,
            static_pn_ratio_equivalence=False,
        )
        config = MultiLabelEnumeratedTyperConfig(
            train_datasets=MSMLC_DATASET_PATH, train_args=train_args,
            model_output_path="data/model/trained_msmlc_model"
        )
        typer = MultiLabelEnumeratedTyper(config)
        typer.train()
