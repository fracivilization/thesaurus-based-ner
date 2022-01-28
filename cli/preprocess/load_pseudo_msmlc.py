from datasets.dataset_dict import DatasetDict
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from src.ner_model.abstract_model import NERModel, NERModelConfig
from src.ner_model.two_stage import TwoStageConfig
from src.ner_model.chunker.spacy_model import SpacyNPChunkerConfig
from src.ner_model.typer.dict_match_typer import DictMatchTyperConfig
from src.builder import dataset_builder, ner_model_builder
from src.ner_model.multi_label import multi_label_ner_model_builder
import logging
from datasets import Dataset
import os
from src.dataset.pseudo_dataset.pseudo_msmlc_dataset import (
    PseudoMSMLCAnnoConfig,
    load_pseudo_dataset,
    join_pseudo_and_gold_dataset,
)
from hydra.utils import get_original_cwd, to_absolute_path

logger = logging.getLogger(__name__)


from src.ner_model.two_stage import register_two_stage_configs
from src.ner_model.multi_label.two_stage import register_multi_label_two_stage_configs

cs = ConfigStore.instance()
cs.store(name="base_pseudo_msmlc_anno", node=PseudoMSMLCAnnoConfig)
# cs.store(group="ner_model", name="base_ner_model_config", node=NERModelConfig)
register_multi_label_two_stage_configs(group="multi_label_ner_model")


@hydra.main(config_path="../../conf", config_name="pseudo_anno_msmlc")
def main(cfg: PseudoMSMLCAnnoConfig):
    if not os.path.exists(os.path.join(get_original_cwd(), cfg.output_dir)):
        if cfg.raw_corpus.startswith("data/gold"):
            raw_corpus = DatasetDict.load_from_disk(
                os.path.join(get_original_cwd(), cfg.raw_corpus)
            )["train"]
        elif cfg.raw_corpus.startswith("data/raw"):
            raw_corpus = Dataset.load_from_disk(
                os.path.join(get_original_cwd(), cfg.raw_corpus)
            )
            assert not cfg.remove_fp_instance
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
    ner_tags[42], ret_datasets["train"][42]


if __name__ == "__main__":
    main()
