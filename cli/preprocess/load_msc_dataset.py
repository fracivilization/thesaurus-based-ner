from datasets.dataset_dict import DatasetDict
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from src.ner_model.abstract_model import NERModel, NERModelConfig
from src.ner_model.two_stage import TwoStageConfig
from src.ner_model.chunker.spacy_model import SpacyNPChunkerConfig
from src.ner_model.typer.dict_match_typer import DictMatchTyperConfig
from src.dataset.utils import DatasetConfig
from omegaconf import MISSING, OmegaConf
from src.dataset import dataset_builder
from src.ner_model import ner_model_builder
import logging
from src.ner_model.evaluator import NERTestor
import json
from datasets import Dataset
import os
from src.ner_model.typer.data_translator import (
    MSCConfig,
    ner_datasets_to_span_classification_datasets,
)
from hydra.utils import get_original_cwd, to_absolute_path

logger = logging.getLogger(__name__)


from src.ner_model.two_stage import register_chunker_configs, chunker_builder

cs = ConfigStore.instance()
cs.store(name="base_msc", node=MSCConfig)
register_chunker_configs("chunker")

from hydra.utils import get_original_cwd, to_absolute_path


@hydra.main(config_path="../../conf", config_name="load_msc")
def main(cfg: MSCConfig) -> None:
    output_dir = to_absolute_path(cfg.output_dir)
    chunker = chunker_builder(cfg.chunker)
    ner_dataset = DatasetDict.load_from_disk(to_absolute_path(cfg.ner_dataset))
    msc_dataset = ner_datasets_to_span_classification_datasets(
        ner_dataset, cfg, chunker
    )
    msc_dataset.save_to_disk(output_dir)

    msc_dataset = DatasetDict.load_from_disk(output_dir)


if __name__ == "__main__":
    main()
