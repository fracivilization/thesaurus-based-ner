import click
from datasets import DatasetDict


@click.command()
@click.option("--input-dataset-dict", type=str, help="Input file path")
def print_multi_label_ner(input_dataset_dict):
    """Prints the multi label ner model"""
    multi_label_ner = DatasetDict.load_from_disk(input_dataset_dict)
    pass
    train_dataset = multi_label_ner["train"]
    label_names = train_dataset.features["labels"].feature.feature.names
    for snt in train_dataset:
        tokens = snt["tokens"]
        print(" ".join(tokens))
        for s, e, labels in zip(snt["starts"], snt["ends"], snt["labels"]):
            span = " ".join(tokens[s:e])
            print("\t".join([span, str(s), str(e)] + [label_names[l] for l in labels]))
    train_dataset["labels"]


if __name__ == "__main__":
    print_multi_label_ner()
