from os import write
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from src.ner_model.abstract_model import NERModel, NERModelConfig, NERModel
from src.ner_model.two_stage import TwoStageModel
from src.dataset.utils import DatasetConfig
from omegaconf import MISSING, OmegaConf, DictConfig
from src.builder import dataset_builder, ner_model_builder
import logging
from src.evaluator import NERTestor, NERTestorConfig
from src.ner_model.two_stage import TwoStageModel
from src.ner_model.bert import register_BERT_configs
from src.ner_model.bond import register_BOND_configs
from src.ner_model.two_stage import register_two_stage_configs
from src.evaluator import register_ner_testor_configs
from src.utils.mlflow import MlflowWriter
import json
import os
import sys

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    ner_model: NERModelConfig = MISSING
    dataset: DatasetConfig = MISSING
    testor: NERTestorConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="base_train_config", node=TrainConfig)
cs.store(group="ner_model", name="base_ner_model_config", node=NERModelConfig)
register_BERT_configs()
register_BOND_configs()
register_two_stage_configs()
register_ner_testor_configs()
cs.store(group="dataset", name="base_dataset_config", node=DatasetConfig)


@hydra.main(config_path="../conf", config_name="train_config")
def main(cfg: TrainConfig):
    writer = MlflowWriter(experiment_name="train")
    print("mlflow_run_id: ", writer.run_id)
    writer.log_param("cwd", os.getcwd())
    writer.log_params_from_omegaconf_dict(cfg)
    dataset = dataset_builder(cfg.dataset)
    dataset_config = DictConfig(
        {
            "dataset": {
                key: json.loads(split.info.description)
                for key, split in dataset.items()
            }
        }
    )
    writer.log_params_from_omegaconf_dict(dataset_config)
    ner_model: NERModel = ner_model_builder(cfg.ner_model, dataset, writer)
    ner_model.train()

    if isinstance(ner_model, TwoStageModel):
        chunk = ner_model.chunker
    else:
        chunk = None
    testor = NERTestor(ner_model, dataset, writer, cfg.testor, chunk)
    writer.set_terminated()


if __name__ == "__main__":
    main()
