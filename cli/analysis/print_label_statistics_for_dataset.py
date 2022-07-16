from collections import defaultdict
from email.policy import default
import click
from src.utils.tree_visualize import (
    get_tree_str,
    make_node2count_consistently_with_child2parent,
)
from src.dataset.utils import STchild2parent, tui2ST
from datasets import Dataset, DatasetDict
from seqeval.metrics.sequence_labeling import get_entities


def get_tui2count_from_multi_label_dataset_dict(dataset_dict: DatasetDict):
    tui2count = defaultdict(lambda: 0)
    for key, split in dataset_dict.items():
        names = split.features["labels"].feature.feature.names
        for snt in split:
            for snt_labels in snt["labels"]:
                for label in snt_labels:
                    tui2count[names[label]] += 1
    return tui2count


def get_tui2count_from_dataset_dict(dataset_dict: DatasetDict):
    tui2count = defaultdict(lambda: 0)
    for key, split in dataset_dict.items():
        ner_tags = split.features["ner_tags"].feature.names
        for snt in split:
            for label, _, _ in get_entities([ner_tags[tag] for tag in snt["ner_tags"]]):
                tui2count[label] += 1
    return tui2count


# データ種類ごとに統計量を獲得する
@click.command()
@click.option(
    "--dataset-path",
    help="Multi Label NER Dataset Path",
    # default="data/gold/multi_label_ner",
    # default="data/pseudo/836d102a37fb1efd797b6eeb9069b7fecd7ccf82",  # PSEUDO_MULTI_LABEL_NER_DATA_ON_GOLD
    # default="data/gold/836053d1bd47c7cf824672a17714a61a354af8e9",  # GOLD NER
    # default="data/pseudo/867eeec785a5fc8f7e7c1be571c3beb63b2a5359",  # PSEUDO DATA ON GOLD w/ negative
    default="data/pseudo/714394dc5923b57efefa528d0a2ed71dfb46758f",  # PSEUDO NER ON GODL w/o negative cats
)
def main(dataset_path):
    dataset_dict = DatasetDict.load_from_disk(dataset_path)
    multi_label = False
    if multi_label:
        tui2count = get_tui2count_from_multi_label_dataset_dict(dataset_dict)
    else:
        tui2count = get_tui2count_from_dataset_dict(dataset_dict)
    tui2count = {k.replace("nc-", ""): v for k, v in tui2count.items()}
    ST2count = {tui2ST[tui]: count for tui, count in tui2count.items()}
    for semantic_type in tui2ST.values():
        if semantic_type not in ST2count:
            ST2count[semantic_type] = 0
    if multi_label:
        node2count = ST2count
    else:
        STchild2parent["ROOT"] = 0
        node2count = make_node2count_consistently_with_child2parent(
            STchild2parent, ST2count
        )
    print(get_tree_str(STchild2parent, node2count))


if __name__ == "__main__":
    main()
