from dataclasses import dataclass
from datasets import DatasetDict
from datasets import load_dataset
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name_or_path: str = "conll2003"


def load_dataset(config: DatasetConfig) -> DatasetDict:
    if config.name_or_path in {"conll2003"}:
        dataset = load_dataset(config.name_or_path)
    pass
