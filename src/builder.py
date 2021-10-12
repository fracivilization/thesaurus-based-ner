from src.evaluator import NERTestor
from typing import NewType
from src.ner_model.abstract_model import NERModel, NERModelConfig, NERModelWrapper
from src.dataset.utils import DatasetConfig
from datasets import DatasetDict, load_dataset
from src.ner_model.bert import BERTNERModel
from src.ner_model.bond import BONDNERModel
from src.ner_model.two_stage import TwoStageConfig, TwoStageModel
from src.ner_model.chunker.abstract_model import Chunker, ChunkerConfig
from src.ner_model.chunker.flair_model import FlairNPChunker
from src.ner_model.chunker.spacy_model import BeneparNPChunker, SpacyNPChunker
from src.ner_model.typer.abstract_model import Typer, TyperConfig
from src.ner_model.typer.dict_match_typer import DictMatchTyper
from src.ner_model.typer.inscon_typer import InsconTyper
from datasets import DatasetDict
from logging import getLogger
from hydra.utils import get_original_cwd
import os

logger = getLogger(__name__)


def dataset_builder(config: DatasetConfig) -> DatasetDict:
    if config.name_or_path in {"conll2003"}:
        return load_dataset(config.name_or_path)
    else:
        return DatasetDict.load_from_disk(
            os.path.join(get_original_cwd(), config.name_or_path)
        )


def ner_model_builder(config: NERModelConfig, datasets: DatasetDict = None) -> NERModel:
    if config.ner_model_name == "BERT":
        ner_model = BERTNERModel(datasets, config)
    elif config.ner_model_name == "BOND":
        ner_model = BONDNERModel(datasets, config)
    elif config.ner_model_name == "TwoStage":
        ner_model = TwoStageModel(config, datasets)
    return NERModelWrapper(ner_model, config)


def two_stage_model_builder(config: TwoStageConfig, datasets: DatasetDict = None):
    return TwoStageModel(config, datasets)
    pass
