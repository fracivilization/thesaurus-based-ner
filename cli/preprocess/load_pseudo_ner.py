from datasets.dataset_dict import DatasetDict
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from src.ner_model.abstract_model import NERModel, NERModelConfig
from src.ner_model import ner_model_builder
import logging
from datasets import Dataset
import os
from src.dataset.pseudo_dataset.pseudo_dataset import (
    PseudoAnnoConfig,
    load_pseudo_dataset,
    join_pseudo_and_gold_dataset,
)
from hydra.utils import get_original_cwd

logger = logging.getLogger(__name__)


from src.ner_model.two_stage import register_two_stage_configs

cs = ConfigStore.instance()
cs.store(name="base_pseudo_anno", node=PseudoAnnoConfig)
cs.store(group="ner_model", name="base_ner_model_config", node=NERModelConfig)
register_two_stage_configs()


@hydra.main(config_path="../../conf", config_name="pseudo_anno")
def main(cfg: PseudoAnnoConfig):
    if cfg.raw_corpus.startswith("data/gold"):
        raw_corpus = DatasetDict.load_from_disk(
            os.path.join(get_original_cwd(), cfg.raw_corpus)
        )["train"]
    elif cfg.raw_corpus.startswith("data/raw"):
        raw_corpus = Dataset.load_from_disk(
            os.path.join(get_original_cwd(), cfg.raw_corpus)
        )
    ner_model: NERModel = ner_model_builder(cfg.ner_model)
    pseudo_dataset = load_pseudo_dataset(raw_corpus, ner_model, cfg)
    gold_corpus = DatasetDict.load_from_disk(
        os.path.join(get_original_cwd(), cfg.gold_corpus)
    )
    ret_datasets = join_pseudo_and_gold_dataset(pseudo_dataset, gold_corpus)
    ret_datasets.save_to_disk(os.path.join(get_original_cwd(), cfg.output_dir))

    ret_datasets = DatasetDict.load_from_disk(
        os.path.join(get_original_cwd(), cfg.output_dir)
    )
    names = ret_datasets["train"].features["ner_tags"].feature.names
    ner_tags = [
        [names[tag] for tag in snt] for snt in ret_datasets["train"]["ner_tags"]
    ]


if __name__ == "__main__":
    main()
