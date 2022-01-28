from src.dataset.utils import DatasetConfig
from datasets import DatasetDict, load_dataset
import os
from hydra.utils import get_original_cwd


def dataset_builder(config: DatasetConfig) -> DatasetDict:
    ner_datasets = None
    if config.name_or_path in {"conll2003"}:
        ner_datasets = load_dataset(config.name_or_path)
    else:
        ner_datasets = DatasetDict.load_from_disk(
            os.path.join(get_original_cwd(), config.name_or_path)
        )
    # TODO label namesがdata splitで一貫していることを保証する
    label_names = None
    for key, split in ner_datasets.items():
        other_label_names = split.features["ner_tags"].feature.names
        if label_names:
            assert other_label_names == label_names
        label_names = other_label_names
    return ner_datasets
