from datasets.dataset_dict import DatasetDict
import hydra
from hydra.core.config_store import ConfigStore
import logging
from src.ner_model.multi_label.ml_typer.data_translator import (
    multi_label_ner_datasets_to_multi_span_multi_label_classification_datasets,
    log_label_ratio,
    MSMLCConfig,
)
from hydra.utils import to_absolute_path

logger = logging.getLogger(__name__)


from src.ner_model.two_stage import register_chunker_configs, chunker_builder


cs = ConfigStore.instance()
cs.store(name="base_msmlc", node=MSMLCConfig)
register_chunker_configs("chunker")

from hydra.utils import to_absolute_path


@hydra.main(config_path="../../conf", config_name="load_msmlc")
def main(cfg: MSMLCConfig) -> None:
    output_dir = to_absolute_path(cfg.output_dir)
    chunker = chunker_builder(cfg.chunker)
    ner_dataset = DatasetDict.load_from_disk(
        to_absolute_path(cfg.multi_label_ner_dataset)
    )
    msmlc_dataset = (
        multi_label_ner_datasets_to_multi_span_multi_label_classification_datasets(
            ner_dataset, cfg, chunker
        )
    )
    msmlc_dataset.save_to_disk(output_dir)

    msmlc_dataset = DatasetDict.load_from_disk(output_dir)
    log_label_ratio(msmlc_dataset)


if __name__ == "__main__":
    main()
