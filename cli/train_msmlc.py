from os import write
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from omegaconf import MISSING, OmegaConf, DictConfig
import logging
from src.ner_model.multi_label.ml_typer.testor import MultiLabelTestor
from src.utils.mlflow import MlflowWriter
import json
import os
import sys
from src.ner_model.multi_label.ml_typer.abstract import (
    MultiLabelTyper,
    MultiLabelTyperConfig,
)
from src.ner_model.multi_label.ml_typer import (
    register_multi_label_typer_configs,
    multi_label_typer_builder,
)
from datasets import Dataset, DatasetDict
from hydra.utils import get_original_cwd, to_absolute_path

logger = logging.getLogger(__name__)


@dataclass
class MultiLabelTyperTrainConfig:
    multi_label_typer: MultiLabelTyperConfig = MISSING
    # msmlc_datasets: str = MISSING


cs = ConfigStore.instance()
cs.store(name="base_multi_label_typer_train_config", node=MultiLabelTyperTrainConfig)
register_multi_label_typer_configs("multi_label_typer")


@hydra.main(
    config_path="../conf/ner_model/multi_label_ner_model",
    config_name="train_msmlc_config",
)
def main(cfg: MultiLabelTyperTrainConfig):
    # TODO: できればSingleLabelTyperも使えるようにしたいな...
    writer = MlflowWriter(experiment_name="train_multi_label_typer")
    print("mlflow_run_id: ", writer.run_id)
    writer.log_param("cwd", os.getcwd())
    writer.log_params_from_omegaconf_dict(cfg)
    # dataset = dataset_builder(cfg.dataset)
    # msmlc_datasets = DatasetDict.load_from_disk(to_absolute_path(cfg.msmlc_datasets))
    # if True:
    #     pass
    #     small_train = Dataset.from_dict(
    #         msmlc_datasets["train"][:1], features=msmlc_datasets["train"].features
    #     )
    #     msmlc_datasets = DatasetDict(
    #         {"train": small_train, "validation": small_train, "test": small_train}
    #     )
    # raise NotImplementedError #TODO: load dataset
    # writer.log_params_from_omegaconf_dict(dataset_config)
    ml_typer: MultiLabelTyper = multi_label_typer_builder(cfg.multi_label_typer, writer)
    # TODO: ml_typer builderを作ってロードする
    # 先に辞書マッチ実装してもいいか....MLTyper単体でTestできるならややこしくない(NERとの接続考えずに実装できる)し
    ml_typer.train()

    # testor = NERTestor(ml_typer, dataset, writer, cfg.testor, chunk)
    # TODO: MultiLabelTyper単体でtestをする
    # testor = MultiLabelTestor(ml_typer, msmlc_datasets, writer)
    writer.set_terminated()


if __name__ == "__main__":
    main()
