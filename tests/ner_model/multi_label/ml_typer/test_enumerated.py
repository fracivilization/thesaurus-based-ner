import omegaconf
from src.ner_model.multi_label.ml_typer.enumerated import MultiLabelTyperConfig


def test_load_multi_label_typer_config():
    MultiLabelTyperConfig()


def test_loading_BertForEnumeratedMultiLabelTyper():
    # TODO: BertForEnumeratedMultiLabelTyper を loadできる
    pass


def test_loading_MultiLabelEnumeratedTyper():
    # TODO: MultiLabelEnumeratedTyper を loadできる
    pass
