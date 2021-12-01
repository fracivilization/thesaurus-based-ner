#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datasets import DatasetDict
from src.error_analysis.compare_msc_datasets import MSCDatasetComparer
import click
import click


@click.command()
@click.option("--base-msc-datasets")
@click.option("--focus-msc-datasets")
@click.option("--output-dir")
def main(base_msc_datasets: str, focus_msc_datasets: str, output_dir: str):
    MSCDatasetComparer(
        DatasetDict.load_from_disk(base_msc_datasets)["train"],
        DatasetDict.load_from_disk(focus_msc_datasets)["train"],
        output_dir,
    )


if __name__ == "__main__":
    main()
