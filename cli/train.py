from os import write
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from src.ner_model.abstract_model import NERModel, NERModelConfig, NERModel
from src.dataset.utils import DatasetConfig
from omegaconf import MISSING, OmegaConf, DictConfig
from src.dataset import dataset_builder
import logging
from src.ner_model.evaluator import NERTestor, NERTestorConfig
from src.ner_model.two_stage import TwoStageModel
from src.ner_model import register_ner_model_configs, ner_model_builder
from src.ner_model.evaluator import register_ner_testor_configs
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
register_ner_model_configs("ner_model")
register_ner_testor_configs("testor")
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
