from os import write
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from omegaconf import MISSING, OmegaConf, DictConfig
import logging
from src.ner_model.evaluator import MultiLabelTestor
from src.utils.mlflow import MlflowWriter
import json
import os
import sys
from src.ner_model.ml_typer.abstract import MultiLabelTyper, MultiLabelTyperConfig
from src.ner_model.ml_typer import (
    register_multi_label_typer_configs,
    multi_label_typer_builder,
)
from datasets import DatasetDict
from hydra.utils import get_original_cwd, to_absolute_path

logger = logging.getLogger(__name__)


@dataclass
class MultiLabelTyperTrainConfig:
    multi_label_typer: MultiLabelTyperConfig = MISSING
    datasets: str = MISSING


cs = ConfigStore.instance()
cs.store(name="base_multi_label_typer_train_config", node=MultiLabelTyperTrainConfig)
register_multi_label_typer_configs("multi_label_typer")


@hydra.main(config_path="../conf", config_name="train_msmlc_config")
def main(cfg: MultiLabelTyperTrainConfig):
    # TODO: できればSingleLabelTyperも使えるようにしたいな...
    writer = MlflowWriter(experiment_name="train_multi_label_typer")
    print("mlflow_run_id: ", writer.run_id)
    writer.log_param("cwd", os.getcwd())
    writer.log_params_from_omegaconf_dict(cfg)
    # dataset = dataset_builder(cfg.dataset)
    dataset = DatasetDict.load_from_disk(to_absolute_path(cfg.datasets))
    # raise NotImplementedError #TODO: load dataset
    dataset_config = DictConfig(
        {
            "dataset": {
                key: json.loads(split.info.description)
                for key, split in dataset.items()
            }
        }
    )
    writer.log_params_from_omegaconf_dict(dataset_config)
    ml_typer: MultiLabelTyper = multi_label_typer_builder(
        cfg.multi_label_typer, dataset, writer
    )
    # TODO: ml_typer builderを作ってロードする
    # 先に辞書マッチ実装してもいいか....MLTyper単体でTestできるならややこしくない(NERとの接続考えずに実装できる)し
    ml_typer.train()

    # testor = NERTestor(ml_typer, dataset, writer, cfg.testor, chunk)
    # TODO: MultiLabelTyper単体でtestをする
    testor = MultiLabelTestor(ml_typer, dataset, writer)
    writer.set_terminated()


if __name__ == "__main__":
    main()
