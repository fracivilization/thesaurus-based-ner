import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from omegaconf import MISSING
import logging
from src.utils.mlflow import MlflowWriter
import os
from src.ner_model.multi_label.ml_typer.abstract import (
    MultiLabelTyper,
    MultiLabelTyperConfig,
)
from src.ner_model.multi_label.ml_typer import (
    register_multi_label_typer_configs,
    multi_label_typer_builder,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiLabelTyperTrainConfig:
    multi_label_typer: MultiLabelTyperConfig = MISSING


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
    ml_typer: MultiLabelTyper = multi_label_typer_builder(cfg.multi_label_typer, writer)
    ml_typer.train()

    writer.set_terminated()


if __name__ == "__main__":
    main()
