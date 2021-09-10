from src.ner_model.abstract_model import NERModel, NERModelConfig
from src.dataset.utils import DatasetConfig
from datasets import DatasetDict, load_dataset
from src.ner_model.bert import BERTNERModel
from src.ner_model.bond import BONDNERModel
from src.ner_model.two_stage import TwoStageConfig, TwoStageModel
from src.ner_model.chunker.abstract_model import Chunker, ChunkerConfig
from src.ner_model.chunker.flair_model import FlairNPChunker
from src.ner_model.chunker.spacy_model import BeneparNPChunker
from src.ner_model.typer.abstract_model import Typer, TyperConfig
from src.ner_model.typer.dict_match_typer import DictMatchTyper
from src.ner_model.typer.inscon_typer import InsconTyper
from logging import getLogger

logger = getLogger(__name__)


def dataset_builder(config: DatasetConfig) -> DatasetDict:
    if config.name_or_path in {"conll2003"}:
        return load_dataset(config.name_or_path)


def ner_model_builder(config: NERModelConfig, datasets: DatasetDict) -> NERModel:
    if config.ner_model_name == "BERT":
        return BERTNERModel(datasets, config)
    elif config.ner_model_name == "BOND":
        return BONDNERModel(datasets, config)
    elif config.ner_model_name == "TwoStage":
        return two_stage_model_builder(config, datasets)


def chunker_builder(config: ChunkerConfig):
    if config.chunker_name == "FlairNPChunker":
        return FlairNPChunker()
    elif config.chunker_name == "BeneparNPChunker":
        return BeneparNPChunker()
    else:
        raise NotImplementedError


def typer_builder(config: TyperConfig, datasets: DatasetDict):
    if config.typer_name == "DictMatchTyper":
        term2cat = dict()
        logger.info("No term2cat for passing to DictMatchTyper")
        raise NotImplementedError
        return DictMatchTyper(term2cat)
    elif config.typer_name == "Inscon":
        InsconTyper(config, datasets)
    pass


def two_stage_model_builder(config: TwoStageConfig, datasets: DatasetDict):
    chunker: Chunker = chunker_builder(config.chunker)
    typer: Typer = typer_builder(config.typer, datasets)
    TwoStageModel(chunker, typer, datasets)
    pass
