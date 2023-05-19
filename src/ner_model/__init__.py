from .abstract_model import NERModel, NERModelConfig
from datasets import DatasetDict
from src.utils.mlflow import MlflowWriter
from .bert import BERTNERModel, register_BERT_configs
from .bond import BONDNERModel, register_BOND_configs
from .two_stage import TwoStageModel, register_two_stage_configs
from .matcher_model import NERMatcherModel, register_ner_matcher_configs
from .marginal_softmax_model import (
    FlattenMarginalSoftmaxNERModel,
    register_flattern_marginal_softmax_ner_configs,
)
from hydra.core.config_store import ConfigStore


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
    elif config.ner_model_name == "FlattenMarginalSoftmaxNER":
        ner_model = FlattenMarginalSoftmaxNERModel(config)
    else:
        raise NotImplementedError
    return ner_model


def register_ner_model_configs(group="ner_model"):
    cs = ConfigStore()
    cs.store(group="ner_model", name="base_ner_model_config", node=NERModelConfig)
    register_BERT_configs(group)
    register_BOND_configs(group)
    register_two_stage_configs(group)
    register_ner_matcher_configs(group)
    register_flattern_marginal_softmax_ner_configs(group)
