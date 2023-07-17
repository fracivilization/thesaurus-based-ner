import click
from src.dataset.few_shot_sample import few_sample_from_dataset
from datasets import DatasetDict


@click.command()
@click.option("--source-datasetdict", type=str, required=True)
@click.option("--few-shot-num", type=int, required=True)
@click.option("--output-dir", type=str, required=True)
def main(source_datasetdict, few_shot_num, output_dir):
    source_datasetdict = DatasetDict.load_from_disk(source_datasetdict)

    # validation, testはそのまま使う
    ret_datasetdict = {
        "train": few_sample_from_dataset(source_datasetdict["train"], few_shot_num),
        "validation": source_datasetdict["validation"],
        "test": source_datasetdict["test"],
    }
    DatasetDict(ret_datasetdict).save_to_disk(output_dir)


if __name__ == "__main__":
    main()
