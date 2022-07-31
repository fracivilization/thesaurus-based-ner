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
from src.ner_model import ner_model_builder
import logging
from src.ner_model.evaluator import NERTestor
import json
from datasets import Dataset
import os
from src.dataset.pseudo_dataset.pseudo_dataset import (
    PseudoAnnoConfig,
    load_pseudo_dataset,
    join_pseudo_and_gold_dataset,
)
from hydra.utils import get_original_cwd, to_absolute_path

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


if __name__ == "__main__":
    main()

# @click.option(
#     "--raw-corpus",
#     type=str,
#     default="data/raw/adc83b19e793491b1c6ea0fd8b46cd9f32e592fc",
# )
# @click.option("--focus-cats", type=str, default="T116_T126")
# @click.option(
#     "--duplicate-cats",
#     type=str,
#     default="ChemicalSubstance_GeneLocation_Biomolecule_Unknown_TopicalConcept",
# )
# @click.option(
#     "--output-dir",
#     type=str,
#     default="data/pseudo/dbc9968f1dd87e3bfab1fee210700a95ebfa65e1",
# )
# @click.option(
#     "--gold-corpus",
#     type=str,
#     default="data/gold/7bd600d361001d5acc3b1e3f2974b2536027ea20",
# )
