from src.ner_model.abstract_model import NERModel, NERModelConfig
from src.dataset.utils import DatasetConfig
from datasets import DatasetDict, load_dataset
from src.ner_model.bert import BERTNERModel
from src.ner_model.bond import BONDNERModel


def dataset_builder(config: DatasetConfig) -> DatasetDict:
    if config.name_or_path in {"conll2003"}:
        return load_dataset(config.name_or_path)


def ner_model_builder(config: NERModelConfig, datasets: DatasetDict) -> NERModel:
    if config.ner_model_name == "BERT":
        return BERTNERModel(datasets, config)
    elif config.ner_model_name == "BOND":
        return BONDNERModel(datasets, config)
