from src.ner_model.multi_label.ml_typer.enumerated import (
    MultiLabelEnumeratedModelArguments,
    MultiLabelEnumeratedTyperConfig,
    BertForEnumeratedMultiLabelTyper,
    MultiLabelEnumeratedTyper,
)
from transformers import AutoConfig
import pytest


@pytest.fixture
def model_args() -> MultiLabelEnumeratedModelArguments:
    return MultiLabelEnumeratedModelArguments(
        model_name_or_path="dmis-lab/biobert-base-cased-v1.1"
    )


@pytest.fixture
def auto_config(model_args) -> AutoConfig:

    task_name = "ner"
    num_labels = 5
    return AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        cache_dir=model_args.cache_dir,
        model_args=model_args,
    )
    pass


@pytest.fixture
def bert_model(model_args, auto_config) -> BertForEnumeratedMultiLabelTyper:
    negative_under_sampling_ratio = 0.05
    return BertForEnumeratedMultiLabelTyper.from_pretrained(
        model_args,
        negative_under_sampling_ratio=negative_under_sampling_ratio,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=auto_config,
        cache_dir=model_args.cache_dir,
    )


class TestBertForEnumeratedMultiLabelTyper:
    def test_loading_BertForEnumeratedMultiLabelTyper(self, bert_model):
        pass


@pytest.fixture
def multi_label_enumerated_typer_config() -> MultiLabelEnumeratedTyperConfig:
    return MultiLabelEnumeratedTyperConfig(
        train_datasets="fixtures/msmlc_gold_100_times_3"
    )


@pytest.fixture
def multi_label_enumerated_typer(
    multi_label_enumerated_typer_config,
) -> MultiLabelEnumeratedTyper:
    MultiLabelEnumeratedTyper(multi_label_enumerated_typer_config)


class TestMultiLabelEnumeratedTyper:
    def test_loading_MultiLabelEnumeratedTyper(self, multi_label_enumerated_typer):
        pass
