from src.evaluator import NERTestor
from typing import NewType
from src.ner_model.abstract_model import NERModel, NERModelConfig, NERModel
from src.dataset.utils import DatasetConfig
from datasets import DatasetDict, load_dataset
from src.ner_model.bert import BERTNERModel
from src.ner_model.bond import BONDNERModel
from src.ner_model.two_stage import TwoStageConfig, TwoStageModel
from datasets import DatasetDict
from logging import getLogger
from hydra.utils import get_original_cwd
import os

from src.utils.mlflow import MlflowWriter

logger = getLogger(__name__)


def dataset_builder(config: DatasetConfig) -> DatasetDict:
    ner_datasets = None
    if config.name_or_path in {"conll2003"}:
        ner_datasets = load_dataset(config.name_or_path)
    else:
        ner_datasets = DatasetDict.load_from_disk(
            os.path.join(get_original_cwd(), config.name_or_path)
        )
    # TODO label namesがdata splitで一貫していることを保証する
    label_names = None
    for key, split in ner_datasets.items():
        other_label_names = split.features["ner_tags"].feature.names
        if label_names:
            assert other_label_names == label_names
        label_names = other_label_names
    return ner_datasets


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
    return ner_model


def two_stage_model_builder(config: TwoStageConfig, datasets: DatasetDict = None):
    return TwoStageModel(config, datasets)
    pass
