import click
from src.dataset.gold_dataset import load_gold_datasets
import os
from datasets import DatasetDict
import sys


@click.command()
@click.option("--focus-cats", type=str, default="T116_T126")
@click.option("--negative-cats", type=str, default="")
@click.option(
    "--output", type=str, default="data/gold/7bd600d361001d5acc3b1e3f2974b2536027ea20"
)
@click.option("--input-dir", type=str, default="data/gold/multi_label_ner")
@click.option("--train-snt-num", type=int, default=sys.maxsize)
def cmd(
    focus_cats: str, negative_cats: str, output: str, input_dir: str, train_snt_num: int
):
    gold_datasets = load_gold_datasets(
        focus_cats, negative_cats, input_dir, train_snt_num
    )
    gold_datasets.save_to_disk(output)


def main():
    cmd()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    pass
