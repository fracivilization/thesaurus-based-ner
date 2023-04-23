from .abstract_model import MultiLabelNERModel, MultiLabelNERModelConfig
from src.utils.mlflow import MlflowWriter
from .two_stage import MultiLabelTwoStageModel, register_multi_label_two_stage_configs
from hydra.core.config_store import ConfigStore


def multi_label_ner_model_builder(
    config: MultiLabelNERModelConfig,
    writer: MlflowWriter = None,
) -> MultiLabelNERModel:
    if config.multi_label_ner_model_name == "MultiLabelTwoStage":
        ner_model = MultiLabelTwoStageModel(config, writer)
    else:
        raise NotImplementedError
    return ner_model

def register_multi_label_ner_model(group="multi_label_ner_model"):
    register_multi_label_two_stage_configs(group=group)
    pass
