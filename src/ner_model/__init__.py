from .abstract_model import NERModel, NERModelConfig
from datasets import DatasetDict
from src.utils.mlflow import MlflowWriter
from .bert import BERTNERModel
from .bond import BONDNERModel
from .two_stage import TwoStageModel
from .matcher_model import NERMatcherModel
from .flatten_ner_model import (
    FlattenMultiLabelNERModel,
    FlattenMultiLabelNERModelConfig,
)


def ner_model_builder(
    config: NERModelConfig, datasets: DatasetDict = None, writer: MlflowWriter = None
) -> NERModel:
    if config.ner_model_name == "BERT":
        ner_model = BERTNERModel(datasets, config)
    elif config.ner_model_name == "BOND":
        ner_model = BONDNERModel(datasets, config)
    elif config.ner_model_name == "TwoStage":
        ner_model = TwoStageModel(config, datasets, writer)
    elif config.ner_model_name == "NERMatcher":
        ner_model = NERMatcherModel(config)
    elif config.ner_model_name == "FlattenMultiLabelNER":
        ner_model = FlattenMultiLabelNERModel(config)
    else:
        raise NotImplementedError
    return ner_model
