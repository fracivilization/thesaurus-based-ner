from src.ner_model.multi_label.ml_typer.enumerated import (
    MultiLabelEnumeratedModelArguments,
    MultiLabelEnumeratedTyperConfig,
    BertForEnumeratedMultiLabelTyper,
    MultiLabelEnumeratedTyper,
    LabelReconstructionArguments,
)
from transformers import AutoConfig
import pytest


@pytest.fixture
def model_args() -> MultiLabelEnumeratedModelArguments:
    return MultiLabelEnumeratedModelArguments(
        model_name_or_path="dmis-lab/biobert-base-cased-v1.1",
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
    def test_loading(self, bert_model):
        pass

    def test_use_dense_label_by_svd(self):
        # TODO: BertForEnumeratedMultiLabelTyperをSVDで圧縮した値で初期化できるようにする
        # NOTE: BertForEnumeratedMultiLabelTyper内ではW^lの初期化のみしか行わない
        # NOTE: SVDの計算自体はMultiLabelEnumeratedTyper内で行う
        _model_args = model_args(use_dense_label_by_svd=True)
        _bert_model = bert_model()


@pytest.fixture
def multi_label_enumerated_typer_config() -> MultiLabelEnumeratedTyperConfig:
    return MultiLabelEnumeratedTyperConfig(
        train_datasets="fixtures/msmlc_gold_100_times_3"
    )


@pytest.fixture
def multi_label_enumerated_typer(
    multi_label_enumerated_typer_config,
) -> MultiLabelEnumeratedTyper:
    return MultiLabelEnumeratedTyper(multi_label_enumerated_typer_config)


@pytest.fixture
def model_args_with_label_reconstruction(
    model_args: MultiLabelEnumeratedModelArguments,
) -> MultiLabelEnumeratedModelArguments:
    model_args.label_reconstruction_args = LabelReconstructionArguments(100, 0.1)
    return model_args


class TestMultiLabelEnumeratedTyper:
    def test_loading(self, multi_label_enumerated_typer):
        pass

    def test_load_label_reconstruction_linear_map(
        self,
        model_args_with_label_reconstruction: MultiLabelEnumeratedModelArguments,
        multi_label_enumerated_typer: MultiLabelEnumeratedTyper,
    ):
        label_reconstruction_linear_map = (
            multi_label_enumerated_typer.load_label_reconstruction_linear_map(
                model_args_with_label_reconstruction,
                multi_label_enumerated_typer.msml_datasets["train"],
            )
        )
        multi_label_enumerated_typer.msml_datasets["train"]
        label_num = (
            multi_label_enumerated_typer.msml_datasets["train"]
            .features["labels"]
            .feature.feature.num_classes
        )
        assert label_reconstruction_linear_map.shape == (
            label_num,
            model_args_with_label_reconstruction.label_reconstruction_args.latent_representation_dim,
        )
