from datasets.dataset_dict import DatasetDict
import hydra
from hydra.core.config_store import ConfigStore
from src.ner_model.multi_label.abstract_model import MultiLabelNERModel
from src.ner_model.multi_label import (
    multi_label_ner_model_builder,
    register_multi_label_ner_model,
)
from omegaconf import OmegaConf
import logging
from datasets import Dataset
import os
from src.dataset.pseudo_dataset.pseudo_multi_label_ner_dataset import (
    PseudoMSMLCAnnoConfig,
    load_msml_pseudo_dataset,
    join_pseudo_and_gold_dataset,
)
from hydra.utils import get_original_cwd

logger = logging.getLogger(__name__)


cs = ConfigStore.instance()
cs.store(name="base_pseudo_msmlc_anno", node=PseudoMSMLCAnnoConfig)
# cs.store(group="ner_model", name="base_ner_model_config", node=NERModelConfig)
register_multi_label_ner_model(group="multi_label_ner_model")


@hydra.main(config_path="../../conf", config_name="pseudo_anno_msmlc")
def main(cfg: PseudoMSMLCAnnoConfig):
    if cfg.raw_corpus.startswith("data/gold"):
        raw_corpus = DatasetDict.load_from_disk(
            os.path.join(get_original_cwd(), cfg.raw_corpus)
        )["train"]
    elif cfg.raw_corpus.startswith("data/raw"):
        raw_corpus = Dataset.load_from_disk(
            os.path.join(get_original_cwd(), cfg.raw_corpus)
        )
    print("load multi label ner model")
    multi_label_ner_model: MultiLabelNERModel = multi_label_ner_model_builder(
        cfg.multi_label_ner_model
    )
    print("load pseudo dataset")
    pseudo_dataset = load_msml_pseudo_dataset(raw_corpus, multi_label_ner_model, cfg)
    print("load gold corpus")
    if OmegaConf.is_missing(cfg, "gold_corpus"):
        ret_datasets = DatasetDict({"train": pseudo_dataset})
    else:
        # TODO: goldデータセットとの統合は別のスクリプトに切り出したい
        gold_corpus = DatasetDict.load_from_disk(
            os.path.join(get_original_cwd(), cfg.gold_corpus)
        )
        ret_datasets = join_pseudo_and_gold_dataset(pseudo_dataset, gold_corpus)
    ret_datasets.save_to_disk(os.path.join(get_original_cwd(), cfg.output_dir))


if __name__ == "__main__":
    main()
