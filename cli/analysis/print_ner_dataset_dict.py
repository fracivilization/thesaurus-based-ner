import click
from datasets import DatasetDict


@click.command()
@click.option("--input-dataset-dict", type=str, help="Input file path")
def main(input_dataset_dict):
    """Prints the ner dataset dict"""
    ner_dataset_dict = DatasetDict.load_from_disk(input_dataset_dict)
    pass
    train_dataset = ner_dataset_dict["train"]
    label_names = train_dataset.features["ner_tags"].feature.names
    for snt in train_dataset:
        tags = [label_names[tag] for tag in snt["ner_tags"]]
        tokens = snt["tokens"]
        print("\n".join("\t".join(line) for line in zip(tokens, tags)))
    train_dataset["ner_tags"]


if __name__ == "__main__":
    main()
