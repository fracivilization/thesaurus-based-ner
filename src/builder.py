from src.evaluator import NERTestor
from typing import NewType
from src.ner_model.abstract_model import NERModel, NERModelConfig, NERModelWrapper
from src.dataset.utils import DatasetConfig
from datasets import DatasetDict, load_dataset
from src.ner_model.bert import BERTNERModel
from src.ner_model.bond import BONDNERModel
from src.ner_model.two_stage import TwoStageConfig, TwoStageModel
from src.ner_model.matcher_model import NERMatcherModel
from datasets import DatasetDict
from logging import getLogger
from hydra.utils import get_original_cwd
import os

from src.utils.mlflow import MlflowWriter

logger = getLogger(__name__)


def dataset_builder(config: DatasetConfig) -> DatasetDict:
    if config.name_or_path in {"conll2003"}:
        return load_dataset(config.name_or_path)
    else:
        return DatasetDict.load_from_disk(
            os.path.join(get_original_cwd(), config.name_or_path)
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
    return NERModelWrapper(ner_model, config)


def two_stage_model_builder(config: TwoStageConfig, datasets: DatasetDict = None):
    return TwoStageModel(config, datasets)
    pass
