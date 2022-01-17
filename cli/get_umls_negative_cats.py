# -*- coding: utf-8 -*-
from datasets import DatasetDict
import click
from src.dataset.utils import get_umls_negative_cats
from typing import List


@click.command()
@click.option("--focus-cats")
def main(focus_cats: str):
    negative_cats: List[str] = get_umls_negative_cats(focus_cats.split("_"))
    print(" ".join(negative_cats))
    pass


if __name__ == "__main__":
    main()
