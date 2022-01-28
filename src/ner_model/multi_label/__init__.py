from .abstract_model import MultiLabelNERModel, MultiLabelNERModelConfig
from src.utils.mlflow import MlflowWriter
from .two_stage import MultiLabelTwoStageModel


def multi_label_ner_model_builder(
    config: MultiLabelNERModelConfig,
    writer: MlflowWriter = None,
) -> MultiLabelNERModel:
    if config.multi_label_ner_model_name == "MultiLabelTwoStage":
        ner_model = MultiLabelTwoStageModel(config, writer)
    else:
        raise NotImplementedError
    return ner_model
