from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from src.ner_model.abstract_model import NERModel, NERModelConfig, NERModelWrapper
from src.ner_model.two_stage import TwoStageModel
from src.dataset.utils import DatasetConfig
from omegaconf import MISSING, OmegaConf
from src.builder import dataset_builder, ner_model_builder
import logging
from src.evaluator import NERTestor
from src.ner_model.two_stage import TwoStageModel


logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    ner_model: NERModelConfig = MISSING
    dataset: DatasetConfig = MISSING


# @dataclass
# class TrainConfig:
#     model_config: str = "sss"
#     data_config: int = 90

from src.ner_model.bert import register_BERT_configs
from src.ner_model.bond import register_BOND_configs
from src.ner_model.two_stage import register_two_stage_configs
from src.ner_model.matcher_model import register_ner_matcher_configs

cs = ConfigStore.instance()
cs.store(name="base_train_config", node=TrainConfig)
cs.store(group="ner_model", name="base_ner_model_config", node=NERModelConfig)
register_BERT_configs()
register_BOND_configs()
register_two_stage_configs()
register_ner_matcher_configs()
cs.store(group="dataset", name="base_dataset_config", node=DatasetConfig)


@hydra.main(config_path="../conf", config_name="train_config")
def main(cfg: TrainConfig):
    dataset = dataset_builder(cfg.dataset)
    ner_model: NERModelWrapper = ner_model_builder(cfg.ner_model, dataset)
    ner_model.train()

    if isinstance(ner_model.ner_model, TwoStageModel):
        chunk = ner_model.ner_model.chunker
    else:
        chunk = None
    testor = NERTestor(ner_model, dataset, chunk)


if __name__ == "__main__":
    main()
